import logging
import tqdm
import time
from datetime import timedelta
import os
import torch
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
from .utils import *
from losses.ce_loss import CrossEntropyLabelSmooth
from losses.cm import ClusterMemoryAMP

def train_neko_sd_pcl(cfg, model, train_loader, val_loader, cluster_loader,
                      optimizer, scheduler, num_query):
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    iters = model.encoder.num_classes // batch

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("logger")
    logger.info('start training')
    
    model.to(device)
    
    
    meters = {}
    for k in ['total', 'ce', 'pcl', 'ldm', 'ce_acc', 'pcl_acc']:
        meters[k] = AverageMeter()
    xent = CrossEntropyLabelSmooth(model.encoder.num_classes)
    
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    
    
    
    all_start_time = time.monotonic()
    
    
    # ######################################################################
    # Train image encoder
    # ######################################################################
    cnt = 0
    for epoch in range(1, epochs+1):
        for v in meters.values():
            v.reset() # reset each meter
        evaluator.reset()
        scheduler.step(epoch)
        logger.info('Learning rate is changed to {:.2e}'.format(scheduler._get_lr(epoch)[0]))
        
        if epoch >= cfg.SD.DDPM.NOISE_ESTIMATE_BEGIN_EP:
            logger.info('Enable noise estimation loss optimization at current epoch.')
        
        
        # Create PCL memory bank
        model.eval()
        image_features = []
        gt_labels = []
        with torch.no_grad():
            for _, (img, pid, camid, _) in enumerate(tqdm.tqdm(cluster_loader, desc='Extract image features')):
                img = img.to(device)
                target = pid.to(device)
                with amp.autocast(True):
                    out = model.encoder.infer_image(img, after_bn=True)
                image_features.append(out)
                gt_labels.append(target)
        image_features = torch.cat(image_features, dim=0)
        gt_labels = torch.cat(gt_labels, dim=0)
        image_features = image_features.float()
        image_features = F.normalize(image_features, dim=1)
        memory = ClusterMemoryAMP(temp=cfg.PCL.MEMORY_TEMP,
                                  momentum=cfg.PCL.MEMORY_MOMENTUM,
                                  use_hard=cfg.PCL.HARD_MEMORY_UPDATE).to(device)
        memory.features = compute_cluster_centroids(image_features, gt_labels).to(device)
        logger.info(f'PCL memory shape: {memory.features.shape}')
        
        
        model.train()
        tloader = tqdm.tqdm(train_loader, total=len(train_loader))
        all_batch_time = []
        batch_cnt = 0
        for n_iter, (file_path, img, img_vae, vid, camid, _) in enumerate(tloader):
            optimizer.zero_grad()
            img = img.to(device)
            img_vae = img_vae.to(device)
            target = vid.to(device)

            loss_dict = {}
            batch_time = time.monotonic()
            with amp.autocast(True):
                fc_logit, proj_feat_bn, noise_loss = model(img, img_vae, prob_type=cfg.SD.UNET.PROB_TYPE, memory=memory,
                                                           camid=camid if cfg.MODEL.ENABLE_CAM_EMB else None, labels=target,
                                                           epoch=epoch)
                
                loss_dict['ldm'] = noise_loss * cfg.MODEL.DDPM_LOSS_WEIGHT
                loss_dict['ce'] = xent(fc_logit, target) * cfg.MODEL.ID_LOSS_WEIGHT
                loss_dict['pcl'] = memory(proj_feat_bn, target) * cfg.MODEL.PCL_LOSS_WEIGHT
                
                loss = sum([v for v in loss_dict.values()])
                loss_dict['total'] = loss
            
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            
            batch_time = time.monotonic() - batch_time
            all_batch_time.append(batch_time)
            
            # udpate meters
            for k in meters.keys():
                if k == 'ce_acc':
                    acc = (fc_logit.max(1)[1] == target).float().mean()
                    meters[k].update(acc, 1)
                elif k == 'pcl_acc':
                    acc = ((proj_feat_bn @ memory.features.t()).max(1)[1] == target).float().mean()
                    meters[k].update(acc, 1)
                else:
                    meters[k].update(loss_dict[k].item(), img.shape[0])

            torch.cuda.synchronize()
            
            tloader.set_description('Epoch [{}/{}]: Prob type={}, ReID loss={:.4e}, noise loss={:.4e}, ce acc={:.1%}, pcl acc={:.1%}, time={:.4f}s'.format(
                epoch,
                epochs,
                cfg.SD.UNET.PROB_TYPE,
                meters['ce'].avg+meters['pcl'].avg,
                meters['ldm'].avg,
                meters['ce_acc'].avg,
                meters['pcl_acc'].avg,
                batch_time
            ))
                
            cnt += 1
            batch_cnt += 1
            
        logger.info("Epoch {} done.".format(epoch))
        logger.info("Average time cost per batch = {:.4f}s".format(sum(all_batch_time)/len(all_batch_time)))
        
        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage2_{}.pth'.format(epoch)))
            
        if epoch % eval_period == 0:
            model.eval()
            for n_iter, (img, vid, camid, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    feat_bn = model.encoder.infer_image(img, after_bn=cfg.TEST.AFTER_BN)
                    evaluator.update((feat_bn, vid, camid))
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info(f'Image encoder training done in {total_time}.')
    print(cfg.OUTPUT_DIR)
    
