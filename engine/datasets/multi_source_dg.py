# encoding: utf-8
import os.path as osp
from .market1501 import Market1501
from .dukemtmcreid import DukeMTMCreID
from .msmt17_v2 import MSMT17_V2
from .cuhk02 import CUHK02
from .cuhk03_np import CUHK03_NP
from .cuhk_sysu import CUHKSYSU
from .ilids import iLIDS
from .prid import PRID
from .grid import GRID
from .viper import VIPeR
from .bases import BaseImageDataset

class MultiSourceDG(BaseImageDataset):
    """
    Multi-source Domain Generalization dataset.
    Use protocol M+D+MS+C.
    3 for training, leaving 1 for test.
    """

    def __init__(self, root='', verbose=True, all_for_train=False, cuhk_protocol='detected',
                 train_datasets=('market1501', 'dukemtmc', 'msmt17', 'cuhk03'),
                 test_dataset='market1501'):
        super(MultiSourceDG, self).__init__()
        
        # create multi-source training set
        init_pid = 0
        init_camid = 0
        train = []
        for d in train_datasets:
            if d == 'market1501':
                ds = Market1501(root, verbose, pid_begin=init_pid, cam_id_begin=init_camid, all_for_train=all_for_train)
            elif d == 'dukemtmc':
                ds = DukeMTMCreID(root, verbose, pid_begin=init_pid, cam_id_begin=init_camid, all_for_train=all_for_train)
            elif d == 'msmt17':
                ds = MSMT17_V2(root, verbose, pid_begin=init_pid, cam_id_begin=init_camid, all_for_train=all_for_train)
            elif d == 'cuhk03':
                ds = CUHK03_NP(root, verbose, cuhk_protocol, pid_begin=init_pid, cam_id_begin=init_camid, all_for_train=all_for_train)
            else:
                raise ValueError(f'Invalid dataset name {d}!')
            train.extend(ds.train)
            init_pid += ds.num_train_pids
            init_camid += ds.num_train_cams
            
        # create multi-source test set
        if test_dataset == 'market1501':
            ds = Market1501(root, verbose, pid_begin=init_pid, cam_id_begin=init_camid, all_for_train=all_for_train)
        elif test_dataset == 'dukemtmc':
            ds = DukeMTMCreID(root, verbose, pid_begin=init_pid, cam_id_begin=init_camid, all_for_train=all_for_train)
        elif test_dataset == 'msmt17':
            ds = MSMT17_V2(root, verbose, pid_begin=init_pid, cam_id_begin=init_camid, all_for_train=all_for_train)
        elif test_dataset == 'cuhk03':
            ds = CUHK03_NP(root, verbose, cuhk_protocol, pid_begin=init_pid, cam_id_begin=init_camid, all_for_train=all_for_train)
        else:
            raise ValueError(f'Invalid dataset name {d}!')
        query = ds.query
        gallery = ds.gallery
        
        if verbose:
            print("###########################")
            print("=> Multi-source DG loaded #")
            print("###########################")
            self.print_dataset_statistics(train, query, gallery)
            print(f'Train: {train_datasets} -> Test: {test_dataset}')
        
        self.train = train
        self.query = query
        self.gallery = gallery
        
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
        
class ClassicalMultiSourceDG(BaseImageDataset):
    """
    Classical multi-source domain generalization dataset.
    Protocol: MA+D+C2+C3+CS (all sets) -> iLIDS, PRID2011, GRID, VIPeR
    """
    train_datasets = ['market1501', 'dukemtmc', 'cuhk02', 'cuhk03', 'cuhk-sysu']
    
    def __init__(self, root='', verbose=True, all_for_train=False, test_dataset='ilids'):
        super().__init__()
        
        # create multi-source training set
        init_pid = 0
        init_camid = 0
        train = []
        for d in ClassicalMultiSourceDG.train_datasets:
            if d == 'market1501':
                ds = Market1501(root, verbose, pid_begin=init_pid, cam_id_begin=init_camid, all_for_train=all_for_train)
            elif d == 'dukemtmc':
                ds = DukeMTMCreID(root, verbose, pid_begin=init_pid, cam_id_begin=init_camid, all_for_train=all_for_train)
            elif d == 'cuhk02':
                ds = CUHK02(root, verbose, pid_begin=init_pid, cam_id_begin=init_camid, all_for_train=all_for_train)
            elif d == 'cuhk03':
                ds = CUHK03_NP(root, verbose, protocol='detected', pid_begin=init_pid, cam_id_begin=init_camid, all_for_train=all_for_train)
            elif d == 'cuhk-sysu':
                ds = CUHKSYSU(root, verbose, pid_begin=init_pid, cam_id_begin=init_camid, all_for_train=all_for_train)
            else:
                raise ValueError(f'Invalid dataset name {d}!')
            train.extend(ds.train)
            init_pid += ds.num_train_pids
            init_camid += ds.num_train_cams
            
        # create multi-source test set
        if test_dataset == 'ilids':
            ds = iLIDS(root, verbose=verbose)
        elif test_dataset == 'prid':
            ds = PRID(root, verbose=verbose)
        elif test_dataset == 'grid':
            ds = GRID(root, verbose=verbose)
        elif test_dataset == 'viper':
            ds = VIPeR(root, verbose=verbose)
        else:
            raise ValueError(f'Invalid dataset name {test_dataset}!')
        query = ds.query
        gallery = ds.gallery
        
        if verbose:
            print("###########################")
            print("=> Multi-source DG loaded #")
            print("###########################")
            self.print_dataset_statistics(train, query, gallery)
            print(f'Train: {ClassicalMultiSourceDG.train_datasets} -> Test: {test_dataset}')
        
        self.train = sorted(train, key=lambda sample: sample[1])
        self.query = sorted(query, key=lambda sample: sample[1])
        self.gallery = sorted(gallery, key=lambda sample: sample[1])
        
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)