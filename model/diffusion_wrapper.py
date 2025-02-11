import torch
import torch.nn as nn
import torch.nn.functional as F
import loralib as lora
from model.stable_diffusion.ddpm import LatentDiffusion

class DCAC(nn.Module):
    def __init__(self, cfg, encoder):
        """
        The DCAC model definition.
        """
        super().__init__()
        self.encoder = encoder
        self.denoiser = LatentDiffusion(cfg, device='cuda')
        
        print('================= Param. Statistics =================')
        print(f'Image encoder parameters: {sum(p.numel() for p in self.encoder.parameters()):,}')
        print(f'U-Net parameters: {sum(p.numel() for p in self.denoiser.model.parameters()):,}')
        print(f'VAE (encoder) parameters: {sum(p.numel() for p in self.denoiser.autoencoder.encoder.parameters()):,}')
        print(f'VAE (decoder) parameters: {sum(p.numel() for p in self.denoiser.autoencoder.decoder.parameters()):,}')
        
        
        print('================= Fine-tune Options =================')
        self.set_finetune_mode(cfg)
        
        print('================= Pre-trained Param. Options =================')
        if cfg.SD.UNET.PRETRAIN != '':
            self.load_denoiser_params(cfg.SD.UNET.PRETRAIN)
        else:
            print('No pre-trained weights for denoiser are loaded.')
            
        # Initialize ID-wise prompts
        print('================= ID Embeddings =================')
        self.denoiser.model.set_id_wise_prompts(
            num_classes=self.encoder.num_classes if cfg.SD.UNET.PROB_TYPE != 'feature' else self.encoder.image_encoder.output_dim,
            emb_dim=cfg.SD.UNET.CONTEXT_DIM,
            type_name=cfg.SD.UNET.CONDITION_BOTTLENECK_TYPE,
            init_std=cfg.SD.UNET.CONDITION_BOTTLENECK_INIT_STD,
            use_hard_label=cfg.SD.UNET.CONDITION_BOTTLENECK_USE_HARD_LABEL
        )
        self.tau = cfg.SD.UNET.CONDITION_BOTTLENECK_TAU
        self.begin_epoch = cfg.SD.DDPM.NOISE_ESTIMATE_BEGIN_EP
        print(f'ID embedding is intialized with: \n{self.denoiser.model.prompt_emb.state_dict().keys()}')
        print(f'Condition temperature: {self.tau}')
        print(f'Model {self.__class__.__name__} has been initialized.')
        print('========================================')
    
    def load_denoiser_params(self, path):
        """
        Load pre-trained weights for denoising U-Net.
        """
        param_dict = torch.load(path, map_location='cpu')['state_dict']
        cnt = 0
        cnt_ignored = 0
        n_params = len(param_dict)
        
        for k, v in param_dict.items():
            try:
                if k.startswith('model.diffusion_model'):
                    self.denoiser.model.state_dict()[k.replace('model.diffusion_model.','')].copy_(v)
                elif k.startswith('first_stage_model'):
                    self.denoiser.autoencoder.state_dict()[k.replace('first_stage_model.', '')].copy_(v)
                elif k.startswith('model_ema') or k.startswith('cond_stage_model'):
                    cnt_ignored += 1
                    continue # ignore SD EMA model and condition embedder, we only use the latest model
                else:
                    self.denoiser.state_dict()[k].copy_(v) # diffusion params (alpha, beta, etc.)
                cnt += 1
            except KeyError:
                print('================= ERROR =================')
                print(f'Fail to load parameter for key = {k}')
        print(f'Successfully process {cnt+cnt_ignored}({cnt} loaded, {cnt_ignored} ignored)/{n_params} params from {path}')
        
        cnt_lora = 0
        for k, _ in self.named_parameters():
            if 'lora_' in k:
                cnt_lora += 1
        print(f'Detected {cnt_lora} LoRA params in the whole model.')
        
    def set_finetune_mode(self, cfg):
        mode = cfg.SD.FINETUNE_MODE
        print(f'Fine-tuning mode: {mode}')
        
        if mode == 'lora':
            self.set_lora_layer(cfg.SD.UNET.LORA_RANK, cfg.SD.UNET.LORA_ALPHA)
        elif mode == 'all':
            print('Fine-tune all U-Net params.')
        elif mode == 'freeze':
            self.freeze_unet()
        else:
            raise ValueError(f'Invalid finetune mode {cfg.SD.FINETUNE_MODE}!')
        
        self.freeze_autoencoder()
        
    def set_lora_layer(self, rank, alpha):
        self.denoiser.model.apply(lambda m: self.add_lora(m, rank, alpha))
        print('LoRA layers have been set up.')
        
    def add_lora(self, m, r, alpha):
        def get_name(module):
            return module.__class__.__name__
        
        if get_name(m) == 'CrossAttention':
            inner_dim, query_dim = m.to_q.weight.shape
            _, context_dim = m.to_k.weight.shape
            m.to_q = lora.Linear(query_dim, inner_dim, r, lora_alpha=alpha)
            m.to_k = lora.Linear(context_dim, inner_dim, r, lora_alpha=alpha)
            m.to_v = lora.Linear(context_dim, inner_dim, r, lora_alpha=alpha)
            m.to_out[0] = lora.Linear(inner_dim, query_dim, r, lora_alpha=alpha)
        elif get_name(m) == 'FeedForward':
            if get_name(m.net[0]) == 'GEGLU':
                dim_out, dim_in = m.net[0].proj.weight.shape
                m.net[0].proj = lora.Linear(dim_in, dim_out, r, lora_alpha=alpha)
            else:
                inner_dim, dim = m.net[0][0].weight.shape
                dim_out, _ = m.net[2].weight.shape
                m.net[0][0] = lora.Linear(dim, inner_dim, r, lora_alpha=alpha)
                m.net[2] = lora.Linear(inner_dim, dim_out, r, lora_alpha=alpha)
    
    def freeze_unet(self):
        cnt = 0
        for v in self.denoiser.model.parameters():
            v.requires_grad_(False)
            cnt += 1
        print(f'Freeze {cnt} params in U-Net.')
    
    def freeze_autoencoder(self):
        cnt = 0
        for v in self.denoiser.autoencoder.parameters():
            v.requires_grad_(False)
            cnt += 1
        print(f'Freeze {cnt} params in VAE.')
        
    def forward(self, img, img_vae, prob_type='fc', memory=None, camid=None, labels=None, epoch=0, return_feat_before_bn=False):
        fc_logit, proj_feat = self.encoder(img=img, cam_label=camid)
        proj_feat_bn = F.normalize(self.encoder.bnneck(proj_feat), dim=1)
        if prob_type == 'fc':
            probs = fc_logit.div(self.tau).softmax(-1) # [B, N]
        elif prob_type == 'pcl':
            assert memory is not None and proj_feat_bn is not None
            probs = ((proj_feat_bn @ memory.features.t()) / memory.temp).softmax(-1) # [B, N]
        elif prob_type == 'fc+pcl':
            assert memory is not None and proj_feat_bn is not None
            pcl_probs = ((proj_feat_bn @ memory.features.t()) / memory.temp).softmax(-1) # [B, N]
            fc_probs = fc_logit.div(self.tau).softmax(-1) # [B, N]
            probs = (fc_probs + pcl_probs) / 2
        elif prob_type == 'feature':
            probs = proj_feat_bn
        else:
            raise ValueError(f'Invalid probability type {prob_type}')
        
        if epoch >= self.begin_epoch:
            noise_loss = self.denoiser(img_vae, probs, labels=labels)
        else:
            noise_loss = torch.tensor(0.0).type_as(fc_logit)
            
        if return_feat_before_bn:
            return fc_logit, proj_feat, noise_loss
        else:
            return fc_logit, proj_feat_bn, noise_loss