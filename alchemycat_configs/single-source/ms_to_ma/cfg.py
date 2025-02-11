from alchemy_cat.dl_config import Config

# default configs
cfg = Config(caps=('alchemycat_configs/base/model.yaml',
                   'alchemycat_configs/base/input.yaml',
                   'alchemycat_configs/base/test.yaml',
                   'alchemycat_configs/base/data/dataloader.yaml',
                   'alchemycat_configs/base/data/dataset.yaml',
                   'alchemycat_configs/base/solver/adam.yaml',
                   'alchemycat_configs/pcl/pcl.yaml',
                   'alchemycat_configs/stable_diffusion/stable_diffusion.yaml'))

cfg.MODEL.ID_LOSS_WEIGHT = 1.0
cfg.MODEL.PCL_LOSS_WEIGHT = 1.0
cfg.MODEL.DDPM_LOSS_WEIGHT = 1.0

cfg.SD.FINETUNE_MODE = 'lora'
cfg.SD.UNET.LORA_RANK = 32

cfg.DATASETS.NAMES = ('msmt17',)
cfg.DATASETS.EVAL_DATASET = 'market1501'