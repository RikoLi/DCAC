from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os
import os.path as osp
import random
import json
import pickle
import numpy as np
import cv2
import lmdb
import torch
import torchvision.transforms as T
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, return_path=False):
        self.dataset = dataset
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.return_path:
            return img_path, img, pid, camid, trackid
        else:
            return img, pid, camid, trackid
        
class ImageDatasetMultimodal(Dataset):
    def __init__(self, dataset, desc_path, transform=None, return_path=False):
        self.dataset = dataset
        self.desc_dict = self.load_desc(desc_path)
        self.transform = transform
        self.return_path = return_path
        
    def load_desc(self, path):
        with open(path, 'r') as f:
            desc_dict = json.load(f)
        return desc_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)
        
        desc_key = osp.basename(img_path)
        desc = self.desc_dict[desc_key]['full']

        if self.transform is not None:
            img = self.transform(img)

        if self.return_path:
            return img_path, img, desc, pid, camid, trackid
        else:
            return img, desc, pid, camid, trackid
        
class ImageDatasetDiT(Dataset):
    def __init__(self, dataset, vae_feature_path, transform) -> None:
        self.dataset = dataset
        self.transform = transform
        with open(vae_feature_path, 'rb') as f:
            self.vae_feature_dict = pickle.load(f) # fname -> feat_vec
            
    def __len__(self):
        assert len(self.vae_feature_dict) == len(self.dataset), 'Number of VAE features and images should be the same'
        return len(self.dataset)
    
    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)
        vae_feature = self.vae_feature_dict[img_path]

        if self.transform is not None:
            img = self.transform(img)
            
        vae_feature = torch.from_numpy(vae_feature)
            
        return img, vae_feature, pid, camid, trackid
        
class ImageDatasetDDPM(Dataset):
    def __init__(self, dataset, transform=None, ddpm_transform=None, return_path=False):
        self.dataset = dataset
        self.transform = transform
        self.ddpm_transform = ddpm_transform
        self.return_path = return_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.ddpm_transform is not None:
            ddpm_img = self.ddpm_transform(img)
        
        if self.transform is not None:
            img = self.transform(img)
            

        if self.return_path:
            return img_path, img, ddpm_img, pid, camid, trackid
        else:
            return img, ddpm_img, pid, camid, trackid
        
class ImageDatasetDDPMMultimodal(Dataset):
    def __init__(self, dataset, desc_path, transform=None, ddpm_transform=None, return_path=False):
        self.dataset = dataset
        self.desc_dict = self.load_desc(desc_path)
        self.transform = transform
        self.ddpm_transform = ddpm_transform
        self.return_path = return_path
        
    def load_desc(self, path):
        with open(path, 'r') as f:
            desc_dict = json.load(f)
        return desc_dict
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)
        
        desc_key = osp.basename(img_path)
        desc = self.desc_dict[desc_key]['full']

        if self.ddpm_transform is not None:
            ddpm_img = self.ddpm_transform(img)
        
        if self.transform is not None:
            img = self.transform(img)
            

        if self.return_path:
            return img_path, img, ddpm_img, desc, pid, camid, trackid
        else:
            return img, ddpm_img, desc, pid, camid, trackid

class IDAAImageDataset(Dataset):
    def __init__(self, dataset, part_labels_lists, transform=None, return_path=False):
        self.dataset = dataset
        self.transform = transform
        self.return_path = return_path
        
        self.part_labels_lists = []
        temp = [label.cpu().tolist() for label in part_labels_lists]
        for tup in zip(*temp):
            self.part_labels_lists.append(tup)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)
        
        part_id = self.part_labels_lists[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.return_path:
            return img_path, img, pid, camid, trackid, part_id
        else:
            return img, pid, camid, trackid, part_id

    
class PseudoLabelImageDataset(ImageDataset):
    def __init__(self, dataset, transform=None):
        super().__init__(dataset, transform)
        
    def __getitem__(self, index):
        # override to return pseudo ID
        img_path, pid, camid, trackid, pseudo_id = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid, pseudo_id
    
class LMDBImageDataset(Dataset):
    def __init__(self, data_dir, key_path, transform=None):
        super(LMDBImageDataset, self).__init__()
        self.data_dir = data_dir
        self.key_path = key_path
        self.transform = transform
        if not os.path.exists(self.data_dir):
            raise IOError('dataset dir: {} is non-exist'.format(
                            self.data_dir))
        self.load_dataset_infos()
        self.env = None
        

    def load_dataset_infos(self):
        if not os.path.exists(self.key_path):
            raise IOError('key info file: {} is non-exist'.format(
                            self.key_path))
        with open(self.key_path, 'rb') as f:
            data = pickle.load(f)
        self.keys = data['keys']
        if 'pids' in data:
            self.labels = np.array(data['pids'], np.int64)
        elif 'vids' in data:
            self.labels = np.array(data['vids'], np.int64)
        else:
            self.labels = np.zeros(len(self.keys), np.int64)
        self.num_cls = len(set(self.labels))

    def __len__(self):
        return len(self.keys)

    def _init_lmdb(self):
        self.env = lmdb.open(self.data_dir, readonly=True, lock=False, 
                        readahead=False, meminit=False)

    def __getitem__(self, index):
        if self.env is None:
            self._init_lmdb()

        key = self.keys[index]
        label = self.labels[index]

        with self.env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))
        img_flat = np.frombuffer(buf, dtype=np.uint8)
        im = cv2.imdecode(img_flat, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            im = Image.fromarray(im)
            im = self.transform(im)
        else:
            im = im / 255.

        return im, label

    def __repr__(self):
        format_string  = self.__class__.__name__ + '(num_imgs='
        format_string += '{:d}, num_cls={:d})'.format(len(self), self.num_cls)
        return format_string
   
    
    
class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if (self.length is not None):
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)
