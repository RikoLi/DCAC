import os
import re
import copy
import glob
import tqdm
import os.path as osp
import numpy as np
from collections import defaultdict
from PIL import Image
from scipy.io import loadmat
from .bases import BaseImageDataset

class CUHKSYSU(BaseImageDataset):
    """CUHKSYSU.

    This dataset can only be used for model training.

    Reference:
        Xiao et al. End-to-end deep learning for person search.

    URL: `<http://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html>`_
    
    Dataset statistics:
        - identities: 11,934
        - images: 34,574    
    """
    dataset_dir = 'cuhk-sysu'
    
    def __init__(self, root='', verbose=True, pid_begin = 0, cam_id_begin=0, all_for_train=False):
        super(CUHKSYSU, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.data_path = osp.join(self.dataset_dir, 'Image', 'SSM')
        self.annotation_path = osp.join(self.dataset_dir, 'annotation')

        self._check_before_run()
        self.pid_begin = pid_begin
        self.cam_id_begin = cam_id_begin
        self.all_for_train = all_for_train
        
        
        self.processed_dir = osp.join(self.dataset_dir, 'cuhksysu4reid')
        self.processed_dir_train = osp.join(self.processed_dir, 'train')
        self.processed_dir_query = osp.join(self.processed_dir, 'query')
        self.processed_dir_gallery = osp.join(self.processed_dir, 'gallery')
        self.processed_dir_combine = osp.join(self.processed_dir, 'combine')
        required_files_state = [self.processed_dir_train, self.processed_dir_query, self.processed_dir_gallery, self.processed_dir_combine]

        # Load image crops if it have preprocessed for ReID,
        # otherwise start preprocessing
        if not all(map(osp.exists, required_files_state)):
            if not osp.exists(self.processed_dir):
                os.mkdir(self.processed_dir)
            os.mkdir(self.processed_dir_train)
            os.mkdir(self.processed_dir_combine)
            os.mkdir(self.processed_dir_query)
            os.mkdir(self.processed_dir_gallery)
            self.preprocessing()

        train = self._process_dir(self.processed_dir_train, relabel=True)
        _combine = self._process_dir(self.processed_dir_combine, relabel=True)
        query = self._process_dir(self.processed_dir_query, relabel=False)
        gallery = self._process_dir(self.processed_dir_gallery, relabel=False)
        
        if self.all_for_train:
            print('=> Use train+query+gallery splits for training.')
            train = _combine
            query = []
            gallery = []
        else:
            print('=> Use train split for training.')
            
        if verbose:
            print('=> CUHK-SYSU loaded')
            self.print_dataset_statistics(train, query, gallery)
            
        self.train = sorted(train, key=lambda sample: sample[1])
        self.query = sorted(query, key=lambda sample: sample[1])
        self.gallery = sorted(gallery, key=lambda sample: sample[1])

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.data_path):
            raise RuntimeError("'{}' is not available".format(self.data_path))
        if not osp.exists(self.annotation_path):
            raise RuntimeError("'{}' is not available".format(self.annotation_path))
    
    def combine_all(self):
        self.train = self._combine
        self.num_train_pids = self.get_num_pids(self.train)


    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_s([-\d]+)_([-\d]+)_([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, image_name, bbox_index, is_hard = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, _, _, _ = map(int, pattern.search(img_path).groups())
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, self.pid_begin + pid, -1 + self.cam_id_begin, 0)) # dummy cam label -1

        return data

    def crop_store(self, data_dict, save_dir):
        def _crop_store():
            name = save_dir.split('/')[-1]
            image_dict = defaultdict(list)

            index_instance = 0
            for key, person_images in tqdm.tqdm(data_dict.items()):
                for image_path, box, pid_name, pid, im_name, is_hard in person_images:
                    assert osp.exists(image_path)
                    one_img = Image.open(image_path)
                    one_img_copy = copy.deepcopy(one_img)
                    box_tuple = tuple(box.round())
                    box_tuple = tuple(map(int, box_tuple))
                    filled_pid = str(pid).zfill(5)
                    is_hard = str(is_hard)
                    cropped = one_img_copy.crop(box_tuple)
                    image_name = im_name.replace('.jpg', '')
                    cropped_path = osp.join(save_dir,
                                            f'{filled_pid}_{image_name}_{str(index_instance).zfill(7)}_{is_hard}.jpg')
                    cropped.save(cropped_path)
                    image_dict[pid_name].append((cropped_path, int(pid), 0, 'cuhksysu', int(pid)))
                    index_instance = index_instance + 1

            print(f'Finished processing {name} dir!')
            return image_dict

        if not osp.exists(save_dir):
            os.makedirs(save_dir)
            _crop_store()
        else:
            _crop_store()


    def preprocessing(self):
        Train_mat = loadmat(osp.join(self.annotation_path, 'test', 'train_test', 'Train.mat'))
        testg50_mat = loadmat(osp.join(self.annotation_path, 'test', 'train_test', 'TestG50.mat'))['TestG50'].squeeze()
        all_imgs_mat = loadmat(osp.join(self.annotation_path, 'Person.mat'))
        # pool_mat = loadmat(osp.join(self.annotation_path, 'pool.mat'))
        id_name_to_pid = {}
        train_pid_dict = defaultdict(list)

        train = Train_mat['Train'].squeeze()
        n_train = 0
        for index, item in enumerate(train):
            pid_name = item[0, 0][0][0]
            pid = int(pid_name[1:])
            id_name_to_pid[pid_name] = pid
            scenes = item[0, 0][2].squeeze()
            for im_name, box, is_hard in scenes:
                im_name = str(im_name[0])
                is_hard = is_hard[0][0]
                box = box.squeeze().astype(np.int32)
                box[2:] += box[:2]
                image_path = osp.join(self.data_path, im_name)
                train_pid_dict[pid_name].append((image_path, box, pid_name, pid, im_name, is_hard))
                n_train = n_train + 1

        probe_pid_dict = defaultdict(list)
        gallery_pid_dict = defaultdict(list)
        n_probe = 0
        n_gallery = 0
        for query, gallery in zip(testg50_mat['Query'], testg50_mat['Gallery']):
            im_name = str(query['imname'][0, 0][0])
            roi = query['idlocate'][0, 0][0].astype(np.int32)
            roi[2:] += roi[:2]
            is_hard = query['ishard'][0, 0][0, 0]
            pid_name = query['idname'][0, 0][0]
            pid = int(pid_name[1:])
            assert pid_name not in id_name_to_pid.keys()
            id_name_to_pid[pid_name] = pid
            # im_name, bbox, is_hard, idname, flipped
            image_path = osp.join(self.data_path, im_name)
            probe_pid_dict[pid_name].append((image_path, roi, pid_name, pid, im_name, is_hard))
            n_probe = n_probe + 1
            gallery = gallery.squeeze()
            for _gallery in gallery:
                _im_name = str(_gallery['imname'][0])
                _roi = _gallery['idlocate'][0].astype(np.int32)
                if _roi.size == 0:
                    continue
                else:
                    _roi[2:] += _roi[:2]
                    _is_hard = _gallery['ishard'][0][0]
                    # _id_name = _gallery['idname'][0]
                    # im_name, bbox, is_hard, idname, flipped
                    _image_path = osp.join(self.data_path, _im_name)
                    gallery_pid_dict[pid_name].append((_image_path, _roi, pid_name, pid, _im_name, _is_hard))
                    n_gallery = n_gallery + 1

        num_total_pid = len(train_pid_dict) + len(probe_pid_dict)

        print(num_total_pid)
        all_image_dict = defaultdict(list)
        all_imgs = all_imgs_mat['Person'].squeeze()

        n = 0
        for id_name, _, scenes in all_imgs:
            pid_name = id_name[0]
            pid = int(pid_name[1:])
            scenes = scenes.squeeze()
            for im_name, box, is_hard in scenes:
                im_name = str(im_name[0])
                is_hard = is_hard[0, 0]
                box = box.squeeze().astype(np.int32)
                box[2:] += box[:2]
                image_path = osp.join(self.data_path, im_name)
                all_image_dict[pid_name].append((image_path, box, pid_name, pid, im_name, is_hard))
                n = n + 1

        print(n)
        print(f'n_train: {n_train}, n_probe: {n_probe}, n_gallery: {n_gallery} n_all:{n}')
        train_dict = self.crop_store(train_pid_dict, osp.join(self.processed_dir, 'train'))
        probe_dict = self.crop_store(probe_pid_dict, osp.join(self.processed_dir, 'query'))
        gallery_dict = self.crop_store(gallery_pid_dict, osp.join(self.processed_dir, 'gallery'))
        all_dict = self.crop_store(all_image_dict, osp.join(self.processed_dir, 'combine'))