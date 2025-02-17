from intialization import init_configs
import torch
import argparse
import tqdm
import time
from alchemy_cat.dl_config import load_config
from model.clip_image_encoder import make_single_grained_model
from engine.datasets.dataloader import make_dataloader_sd, make_val_dataloader
from engine.solvers.optimizers import make_optimizer_dcac
from engine.solvers.schedulers import WarmupMultiStepLR
from engine.dcac_trainers import train_dcac_pcl
from model.diffusion_wrapper import DCAC
from utils.metrics import R1_mAP_eval

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DCAC")
    parser.add_argument("--test", action='store_true', help="Run in evaluation mode with this flag on.")
    parser.add_argument("--log_dir", type=str, default='logs', help='Log file output directory.')
    parser.add_argument("--config_root", type=str, default='alchemycat_configs', help='Root directory of all AlchemyCat configs.')
    parser.add_argument("--config_file", type=str, help='AlchemyCat cfg.py config file path.')
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    
    cfg = load_config(args.config_file, experiments_root=args.log_dir, config_root=args.config_root)
    cfg.unfreeze()
    cfg.OUTPUT_DIR = cfg.rslt_dir
    cfg.freeze()
    
    logger, cfg = init_configs(cfg)
    
    train_loader, val_loader, cluster_loader, num_queries, num_classes, cam_num, view_num = make_dataloader_sd(cfg)
    
    # evaluate on given dataset only in single-source DG
    if len(cfg.DATASETS.NAMES) == 1 and cfg.DATASETS.EVAL_DATASET != '':
        logger.info(f'Use {cfg.DATASETS.EVAL_DATASET} for evaluation.')
        val_loader, num_queries = make_val_dataloader(cfg)
    
    # Create models
    encoder = make_single_grained_model(cfg, num_classes, cam_num, view_num)
    model = DCAC(cfg, encoder)
    
    optimizer = make_optimizer_dcac(cfg, model, use_lora=cfg.SD.FINETUNE_MODE=='lora')
    scheduler = WarmupMultiStepLR(optimizer, milestones=cfg.SOLVER.STAGE2.STEPS, gamma=cfg.SOLVER.STAGE2.GAMMA,
                                warmup_factor=cfg.SOLVER.STAGE2.WARMUP_FACTOR,
                                warmup_iters=cfg.SOLVER.STAGE2.WARMUP_ITERS,
                                warmup_method=cfg.SOLVER.STAGE2.WARMUP_METHOD)
 
    if not args.test:
        logger.info('Current mode: training')
        train_dcac_pcl(
            cfg,
            model,
            train_loader,
            val_loader,
            cluster_loader,
            optimizer,
            scheduler,
            num_queries
        )
    else:
        logger.info('Current mode: testing')
        param_dict = torch.load(cfg.TEST.WEIGHT)
        model.load_state_dict(param_dict)
        logger.info(f'load weight: {cfg.TEST.WEIGHT}')
        device = 'cuda'
        evaluator = R1_mAP_eval(num_queries, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)
        evaluator.reset()
        model.to(device)
        model.eval()
        
        all_batch_time = []
        batch_cnt = 0
        
        for n_iter, (img, vid, camid, _) in enumerate(tqdm.tqdm(val_loader, desc='Extract features')):
            batch_time = time.monotonic()
            with torch.no_grad():
                img = img.to(device)
                feat_bn = model.encoder.infer_image(img, after_bn=cfg.TEST.AFTER_BN)
                batch_time = time.monotonic() - batch_time
                all_batch_time.append(batch_time)
                batch_cnt += 1
                evaluator.update((feat_bn, vid, camid))
        logger.info("Average time cost per batch = {:.4f}s".format(sum(all_batch_time)/len(all_batch_time)))
        
        cmc, mAP, *_ = evaluator.compute()
        logger.info('Validation Results:')
        logger.info('mAP: {:.1%}'.format(mAP))
        for r in [1, 5, 10]:
            logger.info('CMC curve, Rank-1{:<3}:{:.1%}'.format(r, cmc[r - 1]))
