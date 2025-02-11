import torch
import loralib as lora

def make_optimizer_neko_sd(cfg, model, use_lora=False):
    params = []
    
    # Only mark LoRA layers in UNet first
    if use_lora:
        lora.mark_only_lora_as_trainable(model.denoiser.model)
        
    # But keep ID-wise prompt and condition bottleneck trainable
    cnt = 0
    for key, value in model.denoiser.model.named_parameters():
        if 'prompt_emb' in key or 'condition_bottleneck' in key:
            value.requires_grad_(True)
            cnt += 1
    print(f'Recover {cnt} params as trainable in ID embedding and condition bottleneck.')
    
    # Other params
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if 'prompt_emb' in key or 'lora_' in key:
            lr = cfg.SOLVER.STAGE2.BASE_LR * cfg.SD.UNET.LR_MULT
        else:
            lr = cfg.SOLVER.STAGE2.BASE_LR
        weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    
    if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE2.MOMENTUM)
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'Adam':
        optimizer = torch.optim.Adam(params)
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)

    return optimizer