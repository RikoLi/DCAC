import glob
import os.path as osp
from scipy.io import loadmat
from .bases import BaseImageDataset
from .utils import read_json, write_json

class GRID(BaseImageDataset):
    """GRID.

    Reference:
        Loy et al. Multi-camera activity correlation analysis. CVPR 2009.

    URL: `<http://personal.ie.cuhk.edu.hk/~ccloy/downloads_qmul_underground_reid.html>`_
    
    Dataset statistics:
        - identities: 250.
        - images: 1275.
        - cameras: 8.
    """
    dataset_dir = 'grid'
    dataset_url = 'http://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/underground_reid.zip'
    _junk_pids = [0]

    def __init__(self, root='', split_id=0, verbose=True, pid_begin = 0, cam_id_begin=0):
        super(GRID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.probe_path = osp.join(
            self.dataset_dir, 'probe'
        )
        self.gallery_path = osp.join(
            self.dataset_dir, 'gallery'
        )
        self.split_mat_path = osp.join(
            self.dataset_dir, 'features_and_partitions.mat'
        )
        self.split_path = osp.join(self.dataset_dir, 'splits.json')

        self._check_before_run()
        
        self.pid_begin = pid_begin
        self.cam_id_begin = cam_id_begin
        self.prepare_split()

        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, '
                'but expected between 0 and {}'.format(
                    split_id,
                    len(splits) - 1
                )
            )
        split = splits[split_id]

        train = split['train']
        query = split['query']
        gallery = split['gallery']

        train = [tuple(item) for item in train]
        query = [tuple(item) for item in query]
        gallery = [tuple(item) for item in gallery]
        
        if verbose:
            print("=> GRID loaded")
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
        if not osp.exists(self.probe_path):
            raise RuntimeError("'{}' is not available".format(self.probe_path))
        if not osp.exists(self.gallery_path):
            raise RuntimeError("'{}' is not available".format(self.gallery_path))
        if not osp.exists(self.split_mat_path):
            raise RuntimeError("'{}' is not available".format(self.split_mat_path))
    
    def prepare_split(self):
        if not osp.exists(self.split_path):
            print('Creating 10 random splits')
            split_mat = loadmat(self.split_mat_path)
            trainIdxAll = split_mat['trainIdxAll'][0] # length = 10
            probe_img_paths = sorted(
                glob.glob(osp.join(self.probe_path, '*.jpeg'))
            )
            gallery_img_paths = sorted(
                glob.glob(osp.join(self.gallery_path, '*.jpeg'))
            )

            splits = []
            for split_idx in range(10):
                train_idxs = trainIdxAll[split_idx][0][0][2][0].tolist()
                assert len(train_idxs) == 125
                idx2label = {
                    idx: label
                    for label, idx in enumerate(train_idxs)
                }

                train, query, gallery = [], [], []

                # processing probe folder
                for img_path in probe_img_paths:
                    img_name = osp.basename(img_path)
                    img_idx = int(img_name.split('_')[0])
                    camid = int(
                        img_name.split('_')[1]
                    ) - 1 # index starts from 0
                    if img_idx in train_idxs:
                        train.append((img_path, idx2label[img_idx]+self.pid_begin, camid+self.cam_id_begin, 0))
                    else:
                        query.append((img_path, img_idx, camid, 0))

                # process gallery folder
                for img_path in gallery_img_paths:
                    img_name = osp.basename(img_path)
                    img_idx = int(img_name.split('_')[0])
                    camid = int(
                        img_name.split('_')[1]
                    ) - 1 # index starts from 0
                    if img_idx in train_idxs:
                        train.append((img_path, idx2label[img_idx]+self.pid_begin, camid+self.cam_id_begin, 0))
                    else:
                        gallery.append((img_path, img_idx, camid, 0))

                split = {
                    'train': train,
                    'query': query,
                    'gallery': gallery,
                    'num_train_pids': 125,
                    'num_query_pids': 125,
                    'num_gallery_pids': 900
                }
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            write_json(splits, self.split_path)
            print('Split file saved to {}'.format(self.split_path))