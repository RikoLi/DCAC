from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re

from .bases import BaseImageDataset

class MSMT17_V2(BaseImageDataset):
    
    dataset_dir = 'MSMT17'
 
    def __init__(self, root, verbose=True, pid_begin=0, cam_id_begin=0, all_for_train=False, **kwargs):
        super(MSMT17_V2, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        
        self.pid_begin = pid_begin
        self.cam_id_begin = cam_id_begin
        self._check_before_run()

        self.all_for_train = all_for_train
        if self.all_for_train:
            print('=> Use train+query+gallery splits for training.')
            train = self._process_dir([self.train_dir, self.query_dir, self.gallery_dir], relabel=True)
            query = []
            gallery = []
        else:
            print('=> Use train split for training.')
            train = self._process_dir(self.train_dir, relabel=True)
            query = self._process_dir(self.query_dir, relabel=False)
            gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        if isinstance(dir_path, str):
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        elif isinstance(dir_path, list):
            img_paths = []
            for dp in dir_path:
                img_paths.extend(glob.glob(osp.join(dp, '*.jpg')))
            
        #pattern = re.compile(r'([-\d]+)_c(\d)')  # pattern for market and duke
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')  # pattern for msmt17
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            #assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 15
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin+pid, self.cam_id_begin+camid, 0))

        return dataset
