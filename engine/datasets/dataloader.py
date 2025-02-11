import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .market1501 import Market1501
from .msmt17_v2 import MSMT17_V2
from .dukemtmcreid import DukeMTMCreID
from .cuhk03_np import CUHK03_NP
from .multi_source_dg import MultiSourceDG, ClassicalMultiSourceDG
from .preprocessing import RandomErasing
from .dataset import ImageDataset, ImageDatasetDDPM
from .sampler import RandomIdentitySampler

FACTORY = {
    'market1501': Market1501,
    'msmt17': MSMT17_V2,
    'dukemtmc': DukeMTMCreID,
    'cuhk03np': CUHK03_NP,
    'msdg': MultiSourceDG,
    'classical_msdg': ClassicalMultiSourceDG
}

def collate_fn(batch):
    imgs, pids, camids, viewids = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids

def make_val_dataloader(cfg):
    """Only return a dataloader for test split."""
    
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = FACTORY[cfg.DATASETS.EVAL_DATASET](root=cfg.DATASETS.ROOT_DIR)
    val_set = ImageDataset(dataset.query+dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn
    )
    return val_loader, len(dataset.query)

def make_dataloader_sd(cfg):
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = FACTORY[cfg.DATASETS.NAMES[0]](root=cfg.DATASETS.ROOT_DIR) # single-source
    elif len(cfg.DATASETS.NAMES) > 1:
        dataset = FACTORY['msdg'](root=cfg.DATASETS.ROOT_DIR, cuhk_protocol='detected',
                                  train_datasets=cfg.DATASETS.NAMES,
                                  test_dataset=cfg.DATASETS.EVAL_DATASET) # multi-source
    else:
        raise ValueError(f'Invalid dataset input type {type(cfg.DATASETS.NAMES)}!')
    
    num_workers = cfg.DATALOADER.NUM_WORKERS
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    
    # train loader
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])
    ddpm_transform_list = [
        T.Resize((cfg.SD.DDPM.IMAGE_SIZE, cfg.SD.DDPM.IMAGE_SIZE), interpolation=3)
    ]
    if cfg.SD.DDPM.AUG_RANDOM_CROP:
        ddpm_transform_list = [
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomCrop(cfg.SD.DDPM.IMAGE_SIZE)
        ]
    ddpm_transform = T.Compose(ddpm_transform_list+[T.ToTensor(),
                                                    T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
    train_set = ImageDatasetDDPM(dataset.train, train_transforms, ddpm_transform, return_path=True)
    batchsize = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    sampler = RandomIdentitySampler(dataset.train, batchsize, cfg.DATALOADER.NUM_INSTANCE)
    train_loader = DataLoader(
        train_set, batch_size=batchsize,
        sampler=sampler,
        num_workers=num_workers
    )
    
    # val loader
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    val_set = ImageDataset(dataset.query+dataset.gallery, val_transforms)
    num_queries = len(dataset.query)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers
    )
    
    
    # cluster loader
    cluster_set = ImageDataset(dataset.train, transform=val_transforms)
    cluster_loader = DataLoader(
        cluster_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, cluster_loader, num_queries, num_classes, cam_num, view_num