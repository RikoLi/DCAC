import os
import random
import torch
import numpy as np
import argparse
from utils.logger import setup_logger

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_configs(cfg):
    set_seed(cfg.MODEL.SEED)

    logger = setup_logger("logger", cfg.rslt_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    
    logger.info(cfg)

    return logger, cfg