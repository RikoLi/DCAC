import glob
import os.path as osp

from .bases import BaseImageDataset


class CUHK02(BaseImageDataset):
    """CUHK02.

    Reference:
        Li and Wang. Locally Aligned Feature Transforms across Views. CVPR 2013.

    URL: `<http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html>`_
    
    Dataset statistics:
        - 5 camera view pairs each with two cameras
        - 971, 306, 107, 193 and 239 identities from P1 - P5
        - totally 1,816 identities
        - image format is png

    Protocol: Use P1 - P4 for training and P5 for evaluation.

    Note: CUHK01 and CUHK02 overlap.
    """
    dataset_dir = 'cuhk02'
    cam_pairs = ['P1', 'P2', 'P3', 'P4', 'P5']
    test_cam_pair = 'P5'

    def __init__(self, root='', verbose=True, pid_begin=0, cam_id_begin=0, all_for_train=False):
        super(CUHK02, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir, 'Dataset')
        self._check_before_run()
        self.pid_begin = pid_begin
        self.cam_id_begin = cam_id_begin
        self.all_for_train = all_for_train

        if self.all_for_train:
            print('=> Use train+query+gallery splits for training.')
            self.test_cam_pair = 'none' # no test samples
        else:
            print('=> Use train split for training.')
        train, query, gallery = self.get_data_list()
            
        if verbose:
            print("=> CUHK02 loaded")
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
        
    def get_data_list(self):
        num_train_pids, camid = 0, 0
        train, query, gallery = [], [], []

        for cam_pair in self.cam_pairs:
            cam_pair_dir = osp.join(self.dataset_dir, cam_pair)

            cam1_dir = osp.join(cam_pair_dir, 'cam1')
            cam2_dir = osp.join(cam_pair_dir, 'cam2')

            impaths1 = glob.glob(osp.join(cam1_dir, '*.png'))
            impaths2 = glob.glob(osp.join(cam2_dir, '*.png'))

            if cam_pair == self.test_cam_pair:
                # add images to query
                for impath in impaths1:
                    pid = osp.basename(impath).split('_')[0]
                    pid = int(pid)
                    query.append((impath, pid+self.pid_begin, camid+self.cam_id_begin, 0))
                camid += 1

                # add images to gallery
                for impath in impaths2:
                    pid = osp.basename(impath).split('_')[0]
                    pid = int(pid)
                    gallery.append((impath, pid+self.pid_begin, camid+self.cam_id_begin, 0))
                camid += 1

            else:
                pids1 = [
                    osp.basename(impath).split('_')[0] for impath in impaths1
                ]
                pids2 = [
                    osp.basename(impath).split('_')[0] for impath in impaths2
                ]
                pids = set(pids1 + pids2)
                pid2label = {
                    pid: label + num_train_pids
                    for label, pid in enumerate(pids)
                }

                # add images to train from cam1
                for impath in impaths1:
                    pid = osp.basename(impath).split('_')[0]
                    pid = pid2label[pid]
                    train.append((impath, pid+self.pid_begin, camid+self.cam_id_begin, 0))
                camid += 1

                # add images to train from cam2
                for impath in impaths2:
                    pid = osp.basename(impath).split('_')[0]
                    pid = pid2label[pid]
                    train.append((impath, pid+self.pid_begin, camid+self.cam_id_begin, 0))
                camid += 1
                num_train_pids += len(pids)

        return train, query, gallery