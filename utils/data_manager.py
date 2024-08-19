from __future__ import print_function, absolute_import
import os
import glob
import urllib
import tarfile
import os.path as osp
import zipfile

from scipy.io import loadmat
import numpy as np
import logging
import h5py
import math
import re

from utils.utils import mkdir_if_missing, write_json, read_json

"""Dataset classes"""

class DukeMTMCVidReID(object):
    """
    DukeMTMCVidReID
    Reference:
    Wu et al. Exploit the Unknown Gradually: One-Shot Video-Based Person
    Re-Identification by Stepwise Learning. CVPR 2018.
    URL: https://github.com/Yu-Wu/DukeMTMC-VideoReID

    Dataset statistics:
    # identities: 702 (train) + 702 (test)
    # tracklets: 2196 (train)  + 2636 (test)
    """

    def __init__(self,
                 root='/data/baishutao/data/dukemtmc-video',
                 sampling_step=32,
                 min_seq_len=0,
                 verbose=True,
                 *args, **kwargs):
        self.dataset_dir = root
        self.dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-VideoReID.zip'

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self.split_train_dense_json_path = osp.join(self.dataset_dir, 'split_train_dense_{}.json'.format(sampling_step))
        self.split_train_json_path = osp.join(self.dataset_dir, 'split_train.json')
        self.split_query_json_path = osp.join(self.dataset_dir, 'split_query.json')
        self.split_gallery_json_path = osp.join(self.dataset_dir, 'split_gallery.json')

        self.split_train_1stframe_json_path = osp.join(self.dataset_dir, 'split_train_1stframe.json')
        self.split_query_1stframe_json_path = osp.join(self.dataset_dir, 'split_query_1stframe.json')
        self.split_gallery_1stframe_json_path = osp.join(self.dataset_dir, 'split_gallery_1stframe.json')

        self.min_seq_len = min_seq_len
        self._check_before_run()

        train, \
        num_train_tracklets, \
        num_train_pids, \
        num_imgs_train = self._process_dir(
            self.train_dir,
            self.split_train_json_path,
            relabel=True)

        train_dense, \
        num_train_tracklets_dense, \
        num_train_pids_dense, \
        num_imgs_train_dense = self._process_dir(
            self.train_dir,
            self.split_train_dense_json_path,
            relabel=True,
            sampling_step=sampling_step)

        query, \
        num_query_tracklets, \
        num_query_pids, \
        num_imgs_query = self._process_dir(
            self.query_dir,
            self.split_query_json_path,
            relabel=False)
        gallery, \
        num_gallery_tracklets, \
        num_gallery_pids, \
        num_imgs_gallery = self._process_dir(
            self.gallery_dir,
            self.split_gallery_json_path,
            relabel=False)

        print("the number of tracklets under dense sampling for train set: {}".
                    format(num_train_tracklets_dense))

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        if verbose:
            print("=> DukeMTMC-VideoReID loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # tracklets")
            print("  ------------------------------")
            print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
            if sampling_step != 0:
                print("  train_d  | {:5d} | {:8d}".format(num_train_pids_dense, num_train_tracklets_dense))
            print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
            print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
            print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
            print("  ------------------------------")

        if sampling_step!=0:
            self.train = train_dense
        else:
            self.train = train

        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

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

    def _process_dir(self, dir_path, json_path, relabel, sampling_step=0):
        if osp.exists(json_path):
            # print("=> {} generated before, awesome!".format(json_path))
            split = read_json(json_path)
            return split['tracklets'], split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet']

        print("=> Automatically generating split (might take a while for the first time, have a coffe)")
        pdirs = glob.glob(osp.join(dir_path, '*'))  # avoid .DS_Store
        print("Processing {} with {} person identities".format(dir_path, len(pdirs)))

        pid_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        tracklets = []
        num_imgs_per_tracklet = []
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            if relabel: pid = pid2label[pid]
            tdirs = glob.glob(osp.join(pdir, '*'))
            for tdir in tdirs:
                raw_img_paths = glob.glob(osp.join(tdir, '*.jpg'))
                num_imgs = len(raw_img_paths)

                if num_imgs < self.min_seq_len:
                    continue

                num_imgs_per_tracklet.append(num_imgs)
                img_paths = []
                for img_idx in range(num_imgs):
                    # some tracklet starts from 0002 instead of 0001
                    img_idx_name = 'F' + str(img_idx + 1).zfill(4)
                    res = glob.glob(osp.join(tdir, '*' + img_idx_name + '*.jpg'))
                    if len(res) == 0:
                        print("Warn: index name {} in {} is missing, jump to next".format(img_idx_name, tdir))
                        continue
                    img_paths.append(res[0])
                img_name = osp.basename(img_paths[0])
                if img_name.find('_') == -1:
                    # old naming format: 0001C6F0099X30823.jpg
                    camid = int(img_name[5]) - 1
                else:
                    # new naming format: 0001_C6_F0099_X30823.jpg
                    camid = int(img_name[6]) - 1
                img_paths = tuple(img_paths)

                # dense sampling
                num_sampling = len(img_paths)//sampling_step
                if num_sampling == 0:
                    tracklets.append((img_paths, pid, camid))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            tracklets.append((img_paths[idx*sampling_step:], pid, camid))
                        else:
                            tracklets.append((img_paths[idx*sampling_step : (idx+1)*sampling_step], pid, camid))

        num_pids = len(pid_container)
        num_tracklets = len(tracklets)

        print("Saving split to {}".format(json_path))
        split_dict = {'tracklets': tracklets, 'num_tracklets': num_tracklets, 'num_pids': num_pids,
            'num_imgs_per_tracklet': num_imgs_per_tracklet, }
        write_json(split_dict, json_path)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


from .bases import BaseImageDataset
class DukeMTMCreID(BaseImageDataset):
    """
    DukeMTMC-reID
    Reference:
    1. Ristani et al. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. ECCVW 2016.
    2. Zheng et al. Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro. ICCV 2017.
    URL: https://github.com/layumi/DukeMTMC-reID_evaluation

    Dataset statistics:
    # identities: 1404 (train + query)
    # images:16522 (train) + 2228 (query) + 17661 (gallery)
    # cameras: 8
    """
    dataset_dir = 'DukeMTMC-reID'

    def __init__(self, root='/home/haoluo/data', verbose=True, **kwargs):
        super(DukeMTMCreID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip'
        print(self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._download_data()
        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> DukeMTMC-reID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _download_data(self):
        if osp.exists(self.dataset_dir):
            print("This dataset has been downloaded.")
            return

        print("Creating directory {}".format(self.dataset_dir))
        mkdir_if_missing(self.dataset_dir)
        fpath = osp.join(self.dataset_dir, osp.basename(self.dataset_url))

        print("Downloading DukeMTMC-reID dataset")
        urllib.request.urlretrieve(self.dataset_url, fpath)

        print("Extracting files")
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(self.dataset_dir)
        zip_ref.close()

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
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset


class Mars_bak(object):
    """
    MARS

    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.

    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 11310 (gallery)
    # cameras: 6

    Note:
    # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
    # gallery imgs with label=-1 can be remove, which do not influence on final performance.

    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """

    def __init__(self, root=None, min_seq_len=0, split_id=0, *args, **kwargs):
        self._root = root
        self.train_name_path = osp.join(self._root, 'info/train_name.txt')
        self.test_name_path = osp.join(self._root, 'info/test_name.txt')
        self.track_train_info_path = osp.join(self._root, 'info/tracks_train_info.mat')
        self.track_test_info_path = osp.join(self._root, 'info/tracks_test_info.mat')
        self.query_IDX_path = osp.join(self._root, 'info/query_IDX.mat')

        self.sampling_type = split_id
        self._check_before_run()

        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        track_train = loadmat(self.track_train_info_path)['track_train_info']  # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path)['track_test_info']  # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze()  # numpy.ndarray (1980,)
        query_IDX -= 1  # index from 0
        track_query = track_test[query_IDX, :]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX, :]
        # track_gallery = track_test

        train, num_train_tracklets, num_train_pids, num_train_imgs = \
            self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)

        query, num_query_tracklets, num_query_pids, num_query_imgs = \
            self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
            self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

        print("=> MARS loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self._root):
            raise RuntimeError("'{}' is not available".format(self._root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        vids_per_pid_count = np.zeros(len(pid_list))

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue  # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel:
                pid = pid2label[pid]
            camid -= 1  # index starts from 0
            img_names = names[start_index - 1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self._root, home_dir, img_name[:4], img_name) for img_name in img_names]

            if home_dir == 'bbox_train':
                if self.sampling_type == 1250:
                    if vids_per_pid_count[pid] >= 2:
                        continue
                    vids_per_pid_count[pid] = vids_per_pid_count[pid] + 1

                elif self.sampling_type > 0:
                    num_pids = self.sampling_type

                    vids_thred = 2

                    if self.sampling_type == 125:
                        vids_thred = 13

                    if pid >= self.sampling_type: continue

                    if vids_per_pid_count[pid] >= vids_thred:
                        continue
                    vids_per_pid_count[pid] = vids_per_pid_count[pid] + 1
                else:
                    pass

            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

class iLIDSVID_bak(object):
    """
    iLIDS-VID

    Reference:
    Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.
    
    Dataset statistics:
    # identities: 300
    # tracklets: 600
    # cameras: 2

    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
    """

    def __init__(self, root, split_id=0):
        print('Dataset: iLIDSVID spli_id :{}'.format(split_id))

        self.root = root
        self.dataset_url = 'http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar'
        self.data_dir = osp.join(root, 'i-LIDS-VID')
        self.split_dir = osp.join(root, 'train-test people splits')
        self.split_mat_path = osp.join(self.split_dir, 'train_test_splits_ilidsvid.mat')
        self.split_path = osp.join(root, 'splits.json')
        self.cam_1_path = osp.join(root, 'i-LIDS-VID/sequences/cam1')
        self.cam_2_path = osp.join(root, 'i-LIDS-VID/sequences/cam2')

        self._download_data()
        self._check_before_run()

        self._prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> iLIDS-VID loaded w/ split_id {}".format(split_id))
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _download_data(self):
        if osp.exists(self.root):
            # print("This dataset has been downloaded.")
            return

        mkdir_if_missing(self.root)
        fpath = osp.join(self.root, osp.basename(self.dataset_url))

        print("Downloading iLIDS-VID dataset")
        url_opener = urllib.URLopener()
        url_opener.retrieve(self.dataset_url, fpath)

        print("Extracting files")
        tar = tarfile.open(fpath)
        tar.extractall(path=self.root)
        tar.close()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError("'{}' is not available".format(self.split_dir))

    def _prepare_split(self):
        if not osp.exists(self.split_path):
            # print("Creating splits")
            mat_split_data = loadmat(self.split_mat_path)['ls_set']
            
            num_splits = mat_split_data.shape[0]
            num_total_ids = mat_split_data.shape[1]
            assert num_splits == 10
            assert num_total_ids == 300
            num_ids_each = num_total_ids/2

            # pids in mat_split_data are indices, so we need to transform them
            # to real pids
            person_cam1_dirs = os.listdir(self.cam_1_path)
            person_cam2_dirs = os.listdir(self.cam_2_path)

            # make sure persons in one camera view can be found in the other camera view
            assert set(person_cam1_dirs) == set(person_cam2_dirs)

            splits = []
            for i_split in range(num_splits):
                # first 50% for testing and the remaining for training, following Wang et al. ECCV'14.
                train_idxs = sorted(list(mat_split_data[i_split,num_ids_each:]))
                test_idxs = sorted(list(mat_split_data[i_split,:num_ids_each]))
                
                train_idxs = [int(i)-1 for i in train_idxs]
                test_idxs = [int(i)-1 for i in test_idxs]
                
                # transform pids to person dir names
                train_dirs = [person_cam1_dirs[i] for i in train_idxs]
                test_dirs = [person_cam1_dirs[i] for i in test_idxs]
                
                split = {'train': train_dirs, 'test': test_dirs}
                splits.append(split)

            print("Totally {} splits are created, following Wang et al. ECCV'14".format(len(splits)))
            print("Split file is saved to {}".format(self.split_path))
            write_json(splits, self.split_path)

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}

        sampling_step = 0

        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_1_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                # img_names = tuple(img_names)
                pid = dirname2pid[dirname]

                if sampling_step != 0:
                    num_sampling = len(img_names) // sampling_step
                    if num_sampling == 0:
                        tracklets.append((img_names, pid, 0))
                    else:
                        for idx in range(num_sampling):
                            if idx == num_sampling - 1:
                                tracklets.append((img_names[-sampling_step:], pid,0))
                            else:
                                tracklets.append((img_names[idx * sampling_step: (idx + 1) * sampling_step], pid, 0))
                else:
                    tracklets.append((img_names, pid, 0))
                # tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))


            if cam2:
                person_dir = osp.join(self.cam_2_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                # img_names = tuple(img_names)
                pid = dirname2pid[dirname]

                if sampling_step != 0:
                    num_sampling = len(img_names) // sampling_step
                    if num_sampling == 0:
                        tracklets.append((img_names, pid, 1))
                    else:
                        for idx in range(num_sampling):
                            if idx == num_sampling - 1:
                                tracklets.append((img_names[-sampling_step:], pid, 1))
                            else:
                                tracklets.append((img_names[idx * sampling_step: (idx + 1) * sampling_step], pid, 1))
                else:
                    tracklets.append((img_names, pid, 1))
                # tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

class PRID_bak(object):
    """
    PRID

    Reference:
    Hirzer et al. Person Re-Identification by Descriptive and Discriminative Classification. SCIA 2011.
    
    Dataset statistics:
    # identities: 200
    # tracklets: 400
    # cameras: 2

    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """


    def __init__(self, root, split_id=0, min_seq_len=0):

        self.root = root
        self.dataset_url = 'https://files.icg.tugraz.at/f/6ab7e8ce8f/?raw=1'
        self.split_path = osp.join(root, 'splits_prid2011.json')
        self.cam_a_path = osp.join(root, 'multi_shot', 'cam_a')
        self.cam_b_path = osp.join(root, 'multi_shot', 'cam_b')

        self._check_before_run()
        splits = read_json(self.split_path)
        if split_id >=  len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> PRID-2011 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_b_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class Mars(object):
    """
    MARS

    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.

    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 11310 (gallery)
    # cameras: 6

    Note:
    # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
    # gallery imgs with label=-1 can be remove, which do not influence on final performance.

    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """

    def __init__(self, root='/data/datasets/', min_seq_len=0):
        print(root)
        self.root = osp.join(root, 'MARS')
        self.train_name_path = osp.join(self.root, 'info/train_name.txt')
        self.test_name_path = osp.join(self.root, 'info/test_name.txt')
        self.track_train_info_path = osp.join(self.root, 'info/tracks_train_info.mat')
        self.track_test_info_path = osp.join(self.root, 'info/tracks_test_info.mat')
        self.query_IDX_path = osp.join(self.root, 'info/query_IDX.mat')

        self._check_before_run()

        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        track_train = loadmat(self.track_train_info_path)['track_train_info']  # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path)['track_test_info']  # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze()  # numpy.ndarray (1980,)
        query_IDX -= 1  # index from 0
        track_query = track_test[query_IDX, :]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX, :]
        # track_gallery = track_test

        train, num_train_tracklets, num_train_pids, num_train_imgs = \
            self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)

        query, num_query_tracklets, num_query_pids, num_query_imgs = \
            self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
            self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

        print("=> MARS loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue  # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1  # index starts from 0
            img_names = names[start_index - 1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

class Mars_plusLS(object):
    """
    MARS

    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.

    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 11310 (gallery)
    # cameras: 6

    Note:
    # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
    # gallery imgs with label=-1 can be remove, which do not influence on final performance.

    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """

    def __init__(self, root='/data/datasets/', min_seq_len=0):
        print(root)
        self.root = osp.join(root, 'MARS')
        self.train_name_path = osp.join(self.root, 'info/train_name.txt')
        self.test_name_path = osp.join(self.root, 'info/test_name.txt')
        self.track_train_info_path = osp.join(self.root, 'info/tracks_train_info.mat')
        self.track_test_info_path = osp.join(self.root, 'info/tracks_test_info.mat')
        self.query_IDX_path = osp.join(self.root, 'info/query_IDX.mat')

        self._check_before_run()

        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        track_train = loadmat(self.track_train_info_path)['track_train_info']  # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path)['track_test_info']  # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze()  # numpy.ndarray (1980,)
        query_IDX -= 1  # index from 0
        track_query = track_test[query_IDX, :]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX, :]
        # track_gallery = track_test
        ls_dt = LSVID(root='/HDD3/zxq/LS-VID/LS-VID')

        # print(ls_dt._process_data(ls_dt._get_names(ls_dt.train_name_path), json_path=ls_dt.split_train_json_path, relabel=True))
        train_ls,num_train_tracklets_ls,num_train_pids_ls,num_train_imgs_ls = ls_dt._process_data(ls_dt._get_names(ls_dt.train_name_path), json_path=ls_dt.split_train_json_path, relabel=True)
        # print(type(train_ls))
        # for e in train_ls:
        #     print(e)

        modified_train_ls = [[elem[0], elem[1] + 625, elem[2]] for elem in train_ls]

        # print(type(train_ls))
        # for e in modified_train_ls:
        #     print(e)
        # print(num_train_tracklets_ls)
        # print(num_train_pids_ls)
        # print(num_train_imgs_ls)



        train, num_train_tracklets, num_train_pids, num_train_imgs = \
            self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)

        # print(modified_train_ls)

        def convert_to_tuple(element):
            if isinstance(element, list):
                return tuple(convert_to_tuple(sub_element) for sub_element in element)
            else:
                return element

        converted_ls = [tuple(map(convert_to_tuple, sublist)) for sublist in modified_train_ls]

        # print(converted_ls)
        train.extend(converted_ls)
        # print(train)
        # print(result)
        # for e in train:
        #     print(e)

        query, num_query_tracklets, num_query_pids, num_query_imgs = \
            self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
            self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

        print("=> MARS loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids+200, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids+200
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue  # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1  # index starts from 0
            img_names = names[start_index - 1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

class iLIDSVID(object):
    """
    iLIDS-VID

    Reference:
    Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.

    Dataset statistics:
    # identities: 300
    # tracklets: 600
    # cameras: 2

    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
    """
    def __init__(self, root='/data/datasets/', split_id=9):
        self.root = osp.join(root, 'iLIDS-VID')
        self.dataset_url = 'http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar'
        self.data_dir = osp.join(self.root, 'i-LIDS-VID')
        self.split_dir = osp.join(self.root, 'train-test people splits')
        self.split_mat_path = osp.join(self.split_dir, 'train_test_splits_ilidsvid.mat')
        self.split_path = osp.join(self.root, 'splits.json')
        self.cam_1_path = osp.join(self.root, 'i-LIDS-VID/sequences/cam1')
        self.cam_2_path = osp.join(self.root, 'i-LIDS-VID/sequences/cam2')
        # self._download_data()
        self._check_before_run()

        self._prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        # print(train_dirs)
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_data(train_dirs, cam1=True, cam2=True)
        train_dense, _, _, _ = self._process_train_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> iLIDS-VID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.train_dense = train_dense
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _download_data(self):
        if osp.exists(self.root):
            print("This dataset has been downloaded.")
            return

        mkdir_if_missing(self.root)
        fpath = osp.join(self.root, osp.basename(self.dataset_url))

        print("Downloading iLIDS-VID dataset")
        url_opener = urllib.URLopener()
        url_opener.retrieve(self.dataset_url, fpath)

        print("Extracting files")
        tar = tarfile.open(fpath)
        tar.extractall(path=self.root)
        tar.close()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError("'{}' is not available".format(self.split_dir))

    def _prepare_split(self):
        if not osp.exists(self.split_path):
            print("Creating splits")
            mat_split_data = loadmat(self.split_mat_path)['ls_set']

            num_splits = mat_split_data.shape[0]
            num_total_ids = mat_split_data.shape[1]
            assert num_splits == 10
            assert num_total_ids == 300
            num_ids_each = num_total_ids//2

            # pids in mat_split_data are indices, so we need to transform them
            # to real pids
            person_cam1_dirs = os.listdir(self.cam_1_path)
            person_cam2_dirs = os.listdir(self.cam_2_path)

            # make sure persons in one camera view can be found in the other camera view
            assert set(person_cam1_dirs) == set(person_cam2_dirs)

            splits = []
            for i_split in range(num_splits):
                # first 50% for testing and the remaining for training, following Wang et al. ECCV'14.
                train_idxs = sorted(list(mat_split_data[i_split,num_ids_each:]))
                test_idxs = sorted(list(mat_split_data[i_split,:num_ids_each]))

                train_idxs = [int(i)-1 for i in train_idxs]
                test_idxs = [int(i)-1 for i in test_idxs]

                # transform pids to person dir names
                train_dirs = [person_cam1_dirs[i] for i in train_idxs]
                test_dirs = [person_cam1_dirs[i] for i in test_idxs]

                split = {'train': train_dirs, 'test': test_dirs}
                splits.append(split)

            print("Totally {} splits are created, following Wang et al. ECCV'14".format(len(splits)))
            print("Split file is saved to {}".format(self.split_path))
            write_json(splits, self.split_path)

        print("Splits created")

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        # pid2dirname = {pid: dirname for dirname, pid in dirname2pid.items()}
        # print(pid2dirname)

        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_1_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_2_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

    def _process_train_data(self, dirnames, cam1=True, cam2=True, sampling_step=32):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}

        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_1_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                # tracklets.append((img_names, pid, 1))
                # dense sampling
                num_sampling = len(img_names)//sampling_step
                if num_sampling == 0:
                    tracklets.append((img_names, pid, 1))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            tracklets.append((img_names[idx*sampling_step:], pid, 1))
                        else:
                            tracklets.append((img_names[idx*sampling_step : (idx+2)*sampling_step], pid, 1))

                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_2_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                # tracklets.append((img_names, pid, 1))
                # dense sampling
                num_sampling = len(img_names)//sampling_step
                if num_sampling == 0:
                    tracklets.append((img_names, pid, 1))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            tracklets.append((img_names[idx*sampling_step:], pid, 1))
                        else:
                            tracklets.append((img_names[idx*sampling_step : (idx+2)*sampling_step], pid, 1))

                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

class PRID(object):
    """
    PRID

    Reference:
    Hirzer et al. Person Re-Identification by Descriptive and Discriminative Classification. SCIA 2011.

    Dataset statistics:
    # identities: 200
    # tracklets: 400
    # cameras: 2

    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """
    root = '/HDD3/zxq/prid_2011'
    dataset_url = 'https://files.icg.tugraz.at/f/6ab7e8ce8f/?raw=1'
    split_path = osp.join(root, 'splits_prid2011.json')
    cam_a_path = osp.join(root, 'prid_2011', 'multi_shot', 'cam_a')
    cam_b_path = osp.join(root, 'prid_2011', 'multi_shot', 'cam_b')

    def __init__(self, split_id=0, min_seq_len=0, root = '/HDD3/zxq/prid_2011'):
        self._check_before_run()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                "split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits) - 1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
            self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
            self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
            self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> PRID-2011 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname: i for i, dirname in enumerate(dirnames)}

        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_b_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class LSVID(object):
    """
    LS-VID

    Reference:
    Li J, Wang J, Tian Q, Gao W and Zhang S Global-Local Temporal Representations for Video Person Re-Identification[J]. ICCV, 2019

    Dataset statistics:
    # identities: 3772
    # tracklets: 2831 (train) + 3504 (query) + 7829 (gallery)
    # cameras: 15

    Note:
    # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
    # gallery imgs with label=-1 can be remove, which do not influence on final performance.

    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """

    def __init__(self, root=None, sampling_step=48, *args, **kwargs):
        print(root)
        self._root = root

        self.train_name_path = osp.join(self._root, 'list_sequence/list_seq_train.txt')
        self.test_name_path = osp.join(self._root, 'list_sequence/list_seq_test.txt')
        self.query_IDX_path = osp.join(self._root, 'test/data/info_test.mat')
        print(self.train_name_path)
        self._check_before_run()

        # prepare meta data
        track_train = self._get_names(self.train_name_path)
        track_test = self._get_names(self.test_name_path)

        track_train = np.array(track_train)
        track_test = np.array(track_test)

        query_IDX = h5py.File(self.query_IDX_path, mode='r')['query'][0,:]   # numpy.ndarray (1980,)
        query_IDX = np.array(query_IDX, dtype=int)

        query_IDX -= 1  # index from 0
        track_query = track_test[query_IDX, :]

        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX, :]

        self.split_train_dense_json_path = osp.join(self._root,'split_train_dense_{}.json'.format(sampling_step))
        self.split_train_json_path = osp.join(self._root, 'split_train.json')
        self.split_query_json_path = osp.join(self._root, 'split_query.json')
        self.split_gallery_json_path = osp.join(self._root, 'split_gallery.json')

        print(self.split_train_json_path)

        train, num_train_tracklets, num_train_pids, num_train_imgs = \
            self._process_data(track_train, json_path=self.split_train_json_path, relabel=True)

        train_dense, num_train_tracklets_dense, num_train_pids_dense, num_train_imgs_dense = \
            self._process_data(track_train, json_path=self.split_train_dense_json_path, relabel=True, sampling_step=sampling_step)

        query, num_query_tracklets, num_query_pids, num_query_imgs = \
            self._process_data(track_query, json_path=self.split_query_json_path, relabel=False)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
            self._process_data(track_gallery, json_path=self.split_gallery_json_path, relabel=False)

        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

        print("=> LS-VID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        if sampling_step != 0:
            print("  train_d  | {:5d} | {:8d}".format(num_train_pids_dense, num_train_tracklets_dense))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        if sampling_step != 0:
            self.train = train_dense
        else:
            self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self._root):
            raise RuntimeError("'{}' is not available".format(self._root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                basepath, pid = new_line.split(' ')
                names.append([basepath, int(pid)])
        return names

    def _process_data(self,
                      meta_data,
                      relabel=False,
                      json_path=None,
                      sampling_step=48):
        if osp.exists(json_path):
            split = read_json(json_path)
            return split['tracklets'], split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet']

        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 1].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {int(pid): label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        vids_per_pid_count = np.zeros(len(pid_list))

        for tracklet_idx in range(num_tracklets):
            tracklet_path = osp.join(self._root, meta_data[tracklet_idx, 0]) + '*'
            img_paths = glob.glob(tracklet_path)  # avoid .DS_Store
            img_paths.sort()
            pid = int(meta_data[tracklet_idx, 1])
            _, _, camid, _ = osp.basename(img_paths[0]).split('_')[:4]
            camid = int(camid)

            if pid == -1: continue  # junk images are just ignored
            assert 1 <= camid <= 15
            if relabel: pid = pid2label[pid]
            camid -= 1  # index starts from 0
            # print(len(img_paths))
            if(len(img_paths) != 0):
                num_sampling = len(img_paths) // sampling_step
            else:
                continue

            if num_sampling == 0:
                tracklets.append((img_paths, pid, camid))
            else:
                for idx in range(num_sampling):
                    if idx == num_sampling - 1:
                        tracklets.append((img_paths[idx * sampling_step:], pid, camid))
                    else:
                        tracklets.append((img_paths[idx * sampling_step: (idx + 1) * sampling_step], pid, camid))
            num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)

        print("Saving split to {}".format(json_path))
        split_dict = {'tracklets': tracklets, 'num_tracklets': num_tracklets, 'num_pids': num_pids,
            'num_imgs_per_tracklet': num_imgs_per_tracklet, }
        write_json(split_dict, json_path)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

# class LSVID(object):
#     """
#     LS-VID
#
#     Reference:
#     Li J, Wang J, Tian Q, Gao W and Zhang S Global-Local Temporal Representations for Video Person Re-Identification[J]. ICCV, 2019
#
#     Dataset statistics:
#     # identities: 3772
#     # tracklets: 2831 (train) + 3504 (query) + 7829 (gallery)
#     # cameras: 15
#
#     Note:
#     # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
#     # gallery imgs with label=-1 can be remove, which do not influence on final performance.
#
#     Args:
#         min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
#     """
#
#     def __init__(self, root=None, sampling_step=48, *args, **kwargs):
#         self._root = root
#         self.train_name_path = osp.join(self._root, 'list_sequence/list_seq_train.txt')
#         self.test_name_path = osp.join(self._root, 'list_sequence/list_seq_test.txt')
#         self.query_IDX_path = osp.join(self._root, 'test/data/info_test.mat')
#
#         self._check_before_run()
#
#         # prepare meta data
#         track_train = self._get_names(self.train_name_path)
#         track_test = self._get_names(self.test_name_path)
#
#         track_train = np.array(track_train)
#         track_test = np.array(track_test)
#
#         query_IDX = h5py.File(self.query_IDX_path, mode='r')['query'][0,:]   # numpy.ndarray (1980,)
#         query_IDX = np.array(query_IDX, dtype=int)
#
#         query_IDX -= 1  # index from 0
#         track_query = track_test[query_IDX, :]
#
#         gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
#         track_gallery = track_test[gallery_IDX, :]
#
#         self.split_train_dense_json_path = osp.join(self._root,'split_train_dense_{}.json'.format(sampling_step))
#         self.split_train_json_path = osp.join(self._root, 'split_train.json')
#         self.split_query_json_path = osp.join(self._root, 'split_query.json')
#         self.split_gallery_json_path = osp.join(self._root, 'split_gallery.json')
#
#         train, num_train_tracklets, num_train_pids, num_train_imgs = \
#             self._process_data(track_train, json_path=self.split_train_json_path, relabel=True)
#
#         train_dense, num_train_tracklets_dense, num_train_pids_dense, num_train_imgs_dense = \
#             self._process_data(track_train, json_path=self.split_train_dense_json_path, relabel=True, sampling_step=sampling_step)
#
#         query, num_query_tracklets, num_query_pids, num_query_imgs = \
#             self._process_data(track_query, json_path=self.split_query_json_path, relabel=False)
#
#         gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
#             self._process_data(track_gallery, json_path=self.split_gallery_json_path, relabel=False)
#
#         num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
#         min_num = np.min(num_imgs_per_tracklet)
#         max_num = np.max(num_imgs_per_tracklet)
#         avg_num = np.mean(num_imgs_per_tracklet)
#
#         num_total_pids = num_train_pids + num_gallery_pids
#         num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets
#
#         print("=> LS-VID loaded")
#         print("Dataset statistics:")
#         print("  ------------------------------")
#         print("  subset   | # ids | # tracklets")
#         print("  ------------------------------")
#         print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
#         if sampling_step != 0:
#             print("  train_d  | {:5d} | {:8d}".format(num_train_pids_dense, num_train_tracklets_dense))
#         print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
#         print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
#         print("  ------------------------------")
#         print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
#         print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
#         print("  ------------------------------")
#
#         if sampling_step != 0:
#             self.train = train_dense
#         else:
#             self.train = train
#         self.query = query
#         self.gallery = gallery
#
#         self.num_train_pids = num_train_pids
#         self.num_query_pids = num_query_pids
#         self.num_gallery_pids = num_gallery_pids
#
#     def _check_before_run(self):
#         """Check if all files are available before going deeper"""
#         if not osp.exists(self._root):
#             raise RuntimeError("'{}' is not available".format(self._root))
#         if not osp.exists(self.train_name_path):
#             raise RuntimeError("'{}' is not available".format(self.train_name_path))
#         if not osp.exists(self.test_name_path):
#             raise RuntimeError("'{}' is not available".format(self.test_name_path))
#         if not osp.exists(self.query_IDX_path):
#             raise RuntimeError("'{}' is not available".format(self.query_IDX_path))
#
#     def _get_names(self, fpath):
#         names = []
#         with open(fpath, 'r') as f:
#             for line in f:
#                 new_line = line.rstrip()
#                 basepath, pid = new_line.split(' ')
#                 names.append([basepath, int(pid)])
#         return names
#
#     def _process_data(self,
#                       meta_data,
#                       relabel=False,
#                       json_path=None,
#                       sampling_step=0):
#         if osp.exists(json_path):
#             split = read_json(json_path)
#             return split['tracklets'], split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet']
#
#         num_tracklets = meta_data.shape[0]
#         pid_list = list(set(meta_data[:, 1].tolist()))
#         num_pids = len(pid_list)
#
#         if relabel: pid2label = {int(pid): label for label, pid in enumerate(pid_list)}
#         tracklets = []
#         num_imgs_per_tracklet = []
#
#         vids_per_pid_count = np.zeros(len(pid_list))
#
#         for tracklet_idx in range(num_tracklets):
#             tracklet_path = osp.join(self._root, meta_data[tracklet_idx, 0]) + '*'
#             img_paths = glob.glob(tracklet_path)  # avoid .DS_Store
#             img_paths.sort()
#             pid = int(meta_data[tracklet_idx, 1])
#             _, _, camid, _ = osp.basename(img_paths[0]).split('_')[:4]
#             camid = int(camid)
#
#             if pid == -1: continue  # junk images are just ignored
#             assert 1 <= camid <= 15
#             if relabel: pid = pid2label[pid]
#             camid -= 1  # index starts from 0
#
#             num_sampling = len(img_paths) // sampling_step
#             if num_sampling == 0:
#                 tracklets.append((img_paths, pid, camid))
#             else:
#                 for idx in range(num_sampling):
#                     if idx == num_sampling - 1:
#                         tracklets.append((img_paths[idx * sampling_step:], pid, camid))
#                     else:
#                         tracklets.append((img_paths[idx * sampling_step: (idx + 1) * sampling_step], pid, camid))
#             num_imgs_per_tracklet.append(len(img_paths))
#
#         num_tracklets = len(tracklets)
#
#         print("Saving split to {}".format(json_path))
#         split_dict = {'tracklets': tracklets, 'num_tracklets': num_tracklets, 'num_pids': num_pids,
#             'num_imgs_per_tracklet': num_imgs_per_tracklet, }
#         write_json(split_dict, json_path)
#
#         return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class VCCVID(object):
    """ VCCVID

    Reference:

    """

    def __init__(self, root='/data/datasets/', sampling_step=64, seq_len=16, stride=4, **kwargs):

        self.root = osp.join(root, 'VCCVID')
        # print(self.root)
        self.train_path = osp.join(self.root, 'train.txt')
        self.query_path = osp.join(self.root, 'query.txt')
        self.gallery_path = osp.join(self.root, 'gallery.txt')
        self._check_before_run()

        train, num_train_tracklets, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes, _ = \
            self._process_data3(self.train_path, relabel=True)
        clothes2label = self._clothes2label_test(self.query_path, self.gallery_path)
        query, num_query_tracklets, num_query_pids, num_query_imgs, num_query_clothes, _, _ = \
            self._process_data3(self.query_path, relabel=False, clothes2label=clothes2label)
        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs, num_gallery_clothes, _, _ = \
            self._process_data3(self.gallery_path, relabel=False, clothes2label=clothes2label)

        # slice each full-length video in the trainingset into more video clip
        train_dense = self._densesampling_for_trainingset(train, sampling_step)
        # In the test stage, each video sample is divided into a series of equilong video clips with a pre-defined stride.
        recombined_query, query_vid2clip_index = self._recombination_for_testset(query, seq_len=seq_len, stride=stride)
        recombined_gallery, gallery_vid2clip_index = self._recombination_for_testset(gallery, seq_len=seq_len,
                                                                                     stride=stride)

        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_clothes = num_train_clothes + len(clothes2label)
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

        logger = logging.getLogger('reid.dataset')
        logger.info("=> VCCVID loaded")
        logger.info("Dataset statistics:")
        logger.info("  ---------------------------------------------")
        logger.info("  subset       | # ids | # tracklets | # clothes")
        logger.info("  ---------------------------------------------")
        logger.info(
            "  train        | {:5d} | {:11d} | {:9d}".format(num_train_pids, num_train_tracklets, num_train_clothes))
        logger.info(
            "  train_dense  | {:5d} | {:11d} | {:9d}".format(num_train_pids, len(train_dense), num_train_clothes))
        logger.info(
            "  query        | {:5d} | {:11d} | {:9d}".format(num_query_pids, num_query_tracklets, num_query_clothes))
        logger.info("  gallery      | {:5d} | {:11d} | {:9d}".format(num_gallery_pids, num_gallery_tracklets,
                                                                     num_gallery_clothes))
        logger.info("  ---------------------------------------------")
        logger.info(
            "  total        | {:5d} | {:11d} | {:9d}".format(num_total_pids, num_total_tracklets, num_total_clothes))
        logger.info("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        logger.info("  ---------------------------------------------")

        self.train = train
        self.train_dense = train_dense
        self.query = query
        self.gallery = gallery

        self.recombined_query = recombined_query
        self.recombined_gallery = recombined_gallery
        self.query_vid2clip_index = query_vid2clip_index
        self.gallery_vid2clip_index = gallery_vid2clip_index

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_path):
            raise RuntimeError("'{}' is not available".format(self.train_path))
        if not osp.exists(self.query_path):
            raise RuntimeError("'{}' is not available".format(self.query_path))
        if not osp.exists(self.gallery_path):
            raise RuntimeError("'{}' is not available".format(self.gallery_path))

    def _clothes2label_test(self, query_path, gallery_path):
        pid_container = set()
        clothes_container = set()
        with open(query_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        with open(gallery_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        return clothes2label

    def _process_data(self, data_path, relabel=False, clothes2label=None):
        tracklet_path_list = []
        pid_container = set()
        clothes_container = set()
        with open(data_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                tracklet_path_list.append((tracklet_path, pid, clothes_label))
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        if clothes2label is None:
            clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        num_tracklets = len(tracklet_path_list)
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        tracklets = []
        num_imgs_per_tracklet = []
        pid2clothes = np.zeros((num_pids, len(clothes2label)))

        for tracklet_path, pid, clothes_label in tracklet_path_list:
            img_paths = glob.glob(osp.join(self.root, tracklet_path, '*'))
            img_paths.sort()

            clothes = '{}_{}'.format(pid, clothes_label)
            clothes_id = clothes2label[clothes]
            pid2clothes[pid2label[pid], clothes_id] = 1
            if relabel:
                pid = pid2label[pid]
            else:
                pid = int(pid)
            session = tracklet_path.split('/')[0]
            cam = tracklet_path.split('_')[1]
            if session == 'session3':
                camid = int(cam) + 12
            else:
                camid = int(cam)

            num_imgs_per_tracklet.append(len(img_paths))
            tracklets.append((img_paths, pid, camid, clothes_id))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_clothes, pid2clothes, clothes2label

    def _process_data2(self, data_path, relabel=False, clothes2label=None):
        tracklet_path_list = []
        pid_container = set()
        clothes_container = set()
        with open(data_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                tracklet_path_list.append((tracklet_path, pid, clothes_label))
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        if clothes2label is None:
            clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        num_tracklets = len(tracklet_path_list)
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        tracklets = []
        num_imgs_per_tracklet = []
        pid2clothes = np.zeros((num_pids, len(clothes2label)))

        for tracklet_path, pid, clothes_label in tracklet_path_list:
            img_paths = glob.glob(osp.join(self.root, tracklet_path, '*.png'))
            # print("+++++++++++++")
            # print(osp.join(self.root,tracklet_path,'*.jpg'))
            # print(img_paths)
            img_paths.sort()

            clothes = '{}_{}'.format(pid, clothes_label)
            clothes_id = clothes2label[clothes]
            pid2clothes[pid2label[pid], clothes_id] = 1
            if relabel:
                pid = pid2label[pid]
            else:
                pid = int(pid)

            # print("====================")
            # print(tracklet_path)
            camid = tracklet_path.split('/')[-1]

            print(camid)

            # print("====================")
            # print(camid)
            # cam = tracklet_path.split('_')[1]
            # if session == 'session3':
            #     camid = int(cam) + 12
            # else:
            #     camid = int(cam)

            num_imgs_per_tracklet.append(len(img_paths))
            tracklets.append((img_paths, pid, camid, clothes_id))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_clothes, pid2clothes, clothes2label

    def _process_data3(self, data_path, relabel=False, clothes2label=None):
        tracklet_path_list = []
        pid_container = set()
        clothes_container = set()
        with open(data_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                tracklet_path_list.append((tracklet_path, pid, clothes_label))
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        if clothes2label is None:
            clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        num_tracklets = len(tracklet_path_list)
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        tracklets = []
        num_imgs_per_tracklet = []
        pid2clothes = np.zeros((num_pids, len(clothes2label)))

        for tracklet_path, pid, clothes_label in tracklet_path_list:
            img_paths = glob.glob(osp.join(self.root, tracklet_path, '*.jpg'))
            img_paths.sort()

            clothes = '{}_{}'.format(pid, clothes_label)
            clothes_id = clothes2label[clothes]
            pid2clothes[pid2label[pid], clothes_id] = 1
            if relabel:
                pid = pid2label[pid]
            else:
                pid = int(pid)

            camid = tracklet_path.split('/')[-1]

            camid = int(camid)

            num_imgs_per_tracklet.append(len(img_paths))
            tracklets.append((img_paths, pid, camid, clothes_id))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_clothes, pid2clothes, clothes2label

    def _densesampling_for_trainingset(self, dataset, sampling_step=64):
        ''' Split all videos in training set into lots of clips for dense sampling.

        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
            sampling_step (int): sampling step for dense sampling

        Returns:
            new_dataset (list): output dataset
        '''
        new_dataset = []
        for (img_paths, pid, camid, clothes_id) in dataset:
            if sampling_step != 0:
                num_sampling = len(img_paths) // sampling_step
                if num_sampling == 0:
                    new_dataset.append((img_paths, pid, camid, clothes_id))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            new_dataset.append((img_paths[idx * sampling_step:], pid, camid, clothes_id))
                        else:
                            new_dataset.append(
                                (img_paths[idx * sampling_step: (idx + 1) * sampling_step], pid, camid, clothes_id))
            else:
                new_dataset.append((img_paths, pid, camid, clothes_id))

        return new_dataset

    def _recombination_for_testset(self, dataset, seq_len=16, stride=4):
        ''' Split all videos in test set into lots of equilong clips.

        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
            seq_len (int): sequence length of each output clip
            stride (int): temporal sampling stride

        Returns:
            new_dataset (list): output dataset with lots of equilong clips
            vid2clip_index (list): a list contains the start and end clip index of each original video
        '''
        new_dataset = []
        vid2clip_index = np.zeros((len(dataset), 2), dtype=int)
        for idx, (img_paths, pid, camid, clothes_id) in enumerate(dataset):
            # start index
            vid2clip_index[idx, 0] = len(new_dataset)
            # process the sequence that can be divisible by seq_len*stride
            for i in range(len(img_paths) // (seq_len * stride)):
                for j in range(stride):
                    begin_idx = i * (seq_len * stride) + j
                    end_idx = (i + 1) * (seq_len * stride)
                    clip_paths = img_paths[begin_idx: end_idx: stride]
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
            # process the remaining sequence that can't be divisible by seq_len*stride
            if len(img_paths) % (seq_len * stride) != 0:
                # reducing stride
                new_stride = (len(img_paths) % (seq_len * stride)) // seq_len
                for i in range(new_stride):
                    begin_idx = len(img_paths) // (seq_len * stride) * (seq_len * stride) + i
                    end_idx = len(img_paths) // (seq_len * stride) * (seq_len * stride) + seq_len * new_stride
                    clip_paths = img_paths[begin_idx: end_idx: new_stride]
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
                # process the remaining sequence that can't be divisible by seq_len
                if len(img_paths) % seq_len != 0:
                    clip_paths = img_paths[len(img_paths) // seq_len * seq_len:]
                    # loop padding
                    while len(clip_paths) < seq_len:
                        for index in clip_paths:
                            if len(clip_paths) >= seq_len:
                                break
                            clip_paths.append(index)
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
            # end index
            vid2clip_index[idx, 1] = len(new_dataset)
            assert ((vid2clip_index[idx, 1] - vid2clip_index[idx, 0]) == math.ceil(len(img_paths) / seq_len))

        return new_dataset, vid2clip_index.tolist()

class V3DGait(object):
    """ V3DGait

    Reference:

    """

    def __init__(self, root='/data/datasets/', sampling_step=64, seq_len=16, stride=4, **kwargs):

        self.root = osp.join(root, 'V3DGait_VIS')
        # print(self.root)
        self.train_path = osp.join(self.root, 'train.txt')
        self.query_path = osp.join(self.root, 'query.txt')
        self.gallery_path = osp.join(self.root, 'gallery.txt')
        self._check_before_run()

        train, num_train_tracklets, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes, _ = \
            self._process_data3(self.train_path, relabel=True)
        clothes2label = self._clothes2label_test(self.query_path, self.gallery_path)
        query, num_query_tracklets, num_query_pids, num_query_imgs, num_query_clothes, _, _ = \
            self._process_data3(self.query_path, relabel=False, clothes2label=clothes2label)
        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs, num_gallery_clothes, _, _ = \
            self._process_data3(self.gallery_path, relabel=False, clothes2label=clothes2label)

        # slice each full-length video in the trainingset into more video clip
        train_dense = self._densesampling_for_trainingset(train, sampling_step)
        # In the test stage, each video sample is divided into a series of equilong video clips with a pre-defined stride.
        recombined_query, query_vid2clip_index = self._recombination_for_testset(query, seq_len=seq_len, stride=stride)
        recombined_gallery, gallery_vid2clip_index = self._recombination_for_testset(gallery, seq_len=seq_len,
                                                                                     stride=stride)

        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_clothes = num_train_clothes + len(clothes2label)
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

        # logger = logging.getLogger('reid.dataset')
        print("=> V3DGait loaded")
        print("Dataset statistics:")
        print("  ---------------------------------------------")
        print("  subset       | # ids | # tracklets | # clothes")
        print("  ---------------------------------------------")
        print(
            "  train        | {:5d} | {:11d} | {:9d}".format(num_train_pids, num_train_tracklets, num_train_clothes))
        print(
            "  train_dense  | {:5d} | {:11d} | {:9d}".format(num_train_pids, len(train_dense), num_train_clothes))
        print(
            "  query        | {:5d} | {:11d} | {:9d}".format(num_query_pids, num_query_tracklets, num_query_clothes))
        print("  gallery      | {:5d} | {:11d} | {:9d}".format(num_gallery_pids, num_gallery_tracklets,
                                                                     num_gallery_clothes))
        print("  ---------------------------------------------")
        print(
            "  total        | {:5d} | {:11d} | {:9d}".format(num_total_pids, num_total_tracklets, num_total_clothes))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ---------------------------------------------")

        # print(query_vid2clip_index)
        # print(len(query_vid2clip_index))

        self.train = train
        self.train_dense = train_dense
        self.query = query
        self.gallery = gallery

        self.recombined_query = recombined_query
        self.recombined_gallery = recombined_gallery
        self.query_vid2clip_index = query_vid2clip_index
        self.gallery_vid2clip_index = gallery_vid2clip_index

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_path):
            raise RuntimeError("'{}' is not available".format(self.train_path))
        if not osp.exists(self.query_path):
            raise RuntimeError("'{}' is not available".format(self.query_path))
        if not osp.exists(self.gallery_path):
            raise RuntimeError("'{}' is not available".format(self.gallery_path))

    def _clothes2label_test(self, query_path, gallery_path):
        pid_container = set()
        clothes_container = set()
        with open(query_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        with open(gallery_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        return clothes2label

    def _process_data(self, data_path, relabel=False, clothes2label=None):
        tracklet_path_list = []
        pid_container = set()
        clothes_container = set()
        with open(data_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                tracklet_path_list.append((tracklet_path, pid, clothes_label))
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        if clothes2label is None:
            clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        num_tracklets = len(tracklet_path_list)
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        tracklets = []
        num_imgs_per_tracklet = []
        pid2clothes = np.zeros((num_pids, len(clothes2label)))

        for tracklet_path, pid, clothes_label in tracklet_path_list:
            img_paths = glob.glob(osp.join(self.root, tracklet_path, '*'))
            img_paths.sort()

            clothes = '{}_{}'.format(pid, clothes_label)
            clothes_id = clothes2label[clothes]
            pid2clothes[pid2label[pid], clothes_id] = 1
            if relabel:
                pid = pid2label[pid]
            else:
                pid = int(pid)
            session = tracklet_path.split('/')[0]
            cam = tracklet_path.split('_')[1]
            if session == 'session3':
                camid = int(cam) + 12
            else:
                camid = int(cam)

            num_imgs_per_tracklet.append(len(img_paths))
            tracklets.append((img_paths, pid, camid, clothes_id))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_clothes, pid2clothes, clothes2label

    def _process_data2(self, data_path, relabel=False, clothes2label=None):
        tracklet_path_list = []
        pid_container = set()
        clothes_container = set()
        with open(data_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                tracklet_path_list.append((tracklet_path, pid, clothes_label))
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        if clothes2label is None:
            clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        num_tracklets = len(tracklet_path_list)
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        tracklets = []
        num_imgs_per_tracklet = []
        pid2clothes = np.zeros((num_pids, len(clothes2label)))

        for tracklet_path, pid, clothes_label in tracklet_path_list:
            img_paths = glob.glob(osp.join(self.root, tracklet_path, '*.png'))
            # print("+++++++++++++")
            # print(osp.join(self.root,tracklet_path,'*.jpg'))
            # print(img_paths)
            img_paths.sort()

            clothes = '{}_{}'.format(pid, clothes_label)
            clothes_id = clothes2label[clothes]
            pid2clothes[pid2label[pid], clothes_id] = 1
            if relabel:
                pid = pid2label[pid]
            else:
                pid = int(pid)

            # print("====================")
            # print(tracklet_path)
            camid = tracklet_path.split('/')[-1]

            print(camid)

            # print("====================")
            # print(camid)
            # cam = tracklet_path.split('_')[1]
            # if session == 'session3':
            #     camid = int(cam) + 12
            # else:
            #     camid = int(cam)

            num_imgs_per_tracklet.append(len(img_paths))
            tracklets.append((img_paths, pid, camid, clothes_id))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_clothes, pid2clothes, clothes2label

    def _process_data3(self, data_path, relabel=False, clothes2label=None):
        tracklet_path_list = []
        pid_container = set()
        clothes_container = set()
        with open(data_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                tracklet_path_list.append((tracklet_path, pid, clothes_label))
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        if clothes2label is None:
            clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        num_tracklets = len(tracklet_path_list)
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        tracklets = []
        num_imgs_per_tracklet = []
        pid2clothes = np.zeros((num_pids, len(clothes2label)))

        for tracklet_path, pid, clothes_label in tracklet_path_list:
            img_paths = glob.glob(osp.join(self.root, tracklet_path, '*.jpg'))
            img_paths.sort()

            clothes = '{}_{}'.format(pid, clothes_label)
            clothes_id = clothes2label[clothes]
            pid2clothes[pid2label[pid], clothes_id] = 1
            if relabel:
                pid = pid2label[pid]
            else:
                pid = int(pid)

            camid = tracklet_path.split('/')[-1]

            camid = int(camid)

            num_imgs_per_tracklet.append(len(img_paths))
            tracklets.append((img_paths, pid, camid, clothes_id))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_clothes, pid2clothes, clothes2label

    def _densesampling_for_trainingset(self, dataset, sampling_step=64):
        ''' Split all videos in training set into lots of clips for dense sampling.

        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
            sampling_step (int): sampling step for dense sampling

        Returns:
            new_dataset (list): output dataset
        '''
        new_dataset = []
        for (img_paths, pid, camid, clothes_id) in dataset:
            if sampling_step != 0:
                num_sampling = len(img_paths) // sampling_step
                if num_sampling == 0:
                    new_dataset.append((img_paths, pid, camid, clothes_id))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            new_dataset.append((img_paths[idx * sampling_step:], pid, camid, clothes_id))
                        else:
                            new_dataset.append(
                                (img_paths[idx * sampling_step: (idx + 1) * sampling_step], pid, camid, clothes_id))
            else:
                new_dataset.append((img_paths, pid, camid, clothes_id))

        return new_dataset

    def _recombination_for_testset(self, dataset, seq_len=16, stride=4):
        ''' Split all videos in test set into lots of equilong clips.

        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
            seq_len (int): sequence length of each output clip
            stride (int): temporal sampling stride

        Returns:
            new_dataset (list): output dataset with lots of equilong clips
            vid2clip_index (list): a list contains the start and end clip index of each original video
        '''
        new_dataset = []
        vid2clip_index = np.zeros((len(dataset), 2), dtype=int)
        for idx, (img_paths, pid, camid, clothes_id) in enumerate(dataset):
            # start index
            vid2clip_index[idx, 0] = len(new_dataset)
            # process the sequence that can be divisible by seq_len*stride
            for i in range(len(img_paths) // (seq_len * stride)):
                for j in range(stride):
                    begin_idx = i * (seq_len * stride) + j
                    end_idx = (i + 1) * (seq_len * stride)
                    clip_paths = img_paths[begin_idx: end_idx: stride]
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
            # process the remaining sequence that can't be divisible by seq_len*stride
            if len(img_paths) % (seq_len * stride) != 0:
                # reducing stride
                new_stride = (len(img_paths) % (seq_len * stride)) // seq_len
                for i in range(new_stride):
                    begin_idx = len(img_paths) // (seq_len * stride) * (seq_len * stride) + i
                    end_idx = len(img_paths) // (seq_len * stride) * (seq_len * stride) + seq_len * new_stride
                    clip_paths = img_paths[begin_idx: end_idx: new_stride]
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
                # process the remaining sequence that can't be divisible by seq_len
                if len(img_paths) % seq_len != 0:
                    clip_paths = img_paths[len(img_paths) // seq_len * seq_len:]
                    # loop padding
                    while len(clip_paths) < seq_len:
                        for index in clip_paths:
                            if len(clip_paths) >= seq_len:
                                break
                            clip_paths.append(index)
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
            # end index
            vid2clip_index[idx, 1] = len(new_dataset)
            assert ((vid2clip_index[idx, 1] - vid2clip_index[idx, 0]) == math.ceil(len(img_paths) / seq_len))

        return new_dataset, vid2clip_index.tolist()

class V3DGait_extend(object):
    """ V3DGait_extend

    Reference:

    """

    def __init__(self, root='/data/datasets/', sampling_step=64, seq_len=16, stride=4, **kwargs):

        self.root = osp.join(root, 'V3DGait_VIS')
        # print(self.root)
        self.train_path = osp.join(self.root, 'train.txt')
        self.query_path = osp.join(self.root, 'query.txt')
        self.gallery_path = osp.join(self.root, 'gallery.txt')
        self._check_before_run()

        train, num_train_tracklets, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes, _ = \
            self._process_data3(self.train_path, relabel=True)
        clothes2label = self._clothes2label_test(self.query_path, self.gallery_path)
        query, num_query_tracklets, num_query_pids, num_query_imgs, num_query_clothes, _, _ = \
            self._process_data3(self.query_path, relabel=False, clothes2label=clothes2label)
        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs, num_gallery_clothes, _, _ = \
            self._process_data3(self.gallery_path, relabel=False, clothes2label=clothes2label)

        # slice each full-length video in the trainingset into more video clip
        train_dense = self._densesampling_for_trainingset(train, sampling_step)
        # In the test stage, each video sample is divided into a series of equilong video clips with a pre-defined stride.
        recombined_query, query_vid2clip_index = self._recombination_for_testset(query, seq_len=seq_len, stride=stride)
        recombined_gallery, gallery_vid2clip_index = self._recombination_for_testset(gallery, seq_len=seq_len,
                                                                                     stride=stride)

        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_clothes = num_train_clothes + len(clothes2label)
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

        # logger = logging.getLogger('reid.dataset')
        print("=> V3DGait loaded")
        print("Dataset statistics:")
        print("  ---------------------------------------------")
        print("  subset       | # ids | # tracklets | # clothes")
        print("  ---------------------------------------------")
        print(
            "  train        | {:5d} | {:11d} | {:9d}".format(num_train_pids, num_train_tracklets, num_train_clothes))
        print(
            "  train_dense  | {:5d} | {:11d} | {:9d}".format(num_train_pids, len(train_dense), num_train_clothes))
        print(
            "  query        | {:5d} | {:11d} | {:9d}".format(num_query_pids, num_query_tracklets, num_query_clothes))
        print("  gallery      | {:5d} | {:11d} | {:9d}".format(num_gallery_pids, num_gallery_tracklets,
                                                                     num_gallery_clothes))
        print("  ---------------------------------------------")
        print(
            "  total        | {:5d} | {:11d} | {:9d}".format(num_total_pids, num_total_tracklets, num_total_clothes))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ---------------------------------------------")

        # print(query_vid2clip_index)
        # print(len(query_vid2clip_index))

        self.train = train
        self.train_dense = train_dense
        self.query = query
        self.gallery = gallery

        self.recombined_query = recombined_query
        self.recombined_gallery = recombined_gallery
        self.query_vid2clip_index = query_vid2clip_index
        self.gallery_vid2clip_index = gallery_vid2clip_index

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_path):
            raise RuntimeError("'{}' is not available".format(self.train_path))
        if not osp.exists(self.query_path):
            raise RuntimeError("'{}' is not available".format(self.query_path))
        if not osp.exists(self.gallery_path):
            raise RuntimeError("'{}' is not available".format(self.gallery_path))

    def _clothes2label_test(self, query_path, gallery_path):
        pid_container = set()
        clothes_container = set()
        with open(query_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        with open(gallery_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        return clothes2label

    def _process_data(self, data_path, relabel=False, clothes2label=None):
        tracklet_path_list = []
        pid_container = set()
        clothes_container = set()
        with open(data_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                tracklet_path_list.append((tracklet_path, pid, clothes_label))
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        if clothes2label is None:
            clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        num_tracklets = len(tracklet_path_list)
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        tracklets = []
        num_imgs_per_tracklet = []
        pid2clothes = np.zeros((num_pids, len(clothes2label)))

        for tracklet_path, pid, clothes_label in tracklet_path_list:
            img_paths = glob.glob(osp.join(self.root, tracklet_path, '*'))
            img_paths.sort()

            clothes = '{}_{}'.format(pid, clothes_label)
            clothes_id = clothes2label[clothes]
            pid2clothes[pid2label[pid], clothes_id] = 1
            if relabel:
                pid = pid2label[pid]
            else:
                pid = int(pid)
            session = tracklet_path.split('/')[0]
            cam = tracklet_path.split('_')[1]
            if session == 'session3':
                camid = int(cam) + 12
            else:
                camid = int(cam)

            num_imgs_per_tracklet.append(len(img_paths))
            tracklets.append((img_paths, pid, camid, clothes_id))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_clothes, pid2clothes, clothes2label

    def _process_data2(self, data_path, relabel=False, clothes2label=None):
        tracklet_path_list = []
        pid_container = set()
        clothes_container = set()
        with open(data_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                tracklet_path_list.append((tracklet_path, pid, clothes_label))
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        if clothes2label is None:
            clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        num_tracklets = len(tracklet_path_list)
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        tracklets = []
        num_imgs_per_tracklet = []
        pid2clothes = np.zeros((num_pids, len(clothes2label)))

        for tracklet_path, pid, clothes_label in tracklet_path_list:
            img_paths = glob.glob(osp.join(self.root, tracklet_path, '*.png'))
            # print("+++++++++++++")
            # print(osp.join(self.root,tracklet_path,'*.jpg'))
            # print(img_paths)
            img_paths.sort()

            clothes = '{}_{}'.format(pid, clothes_label)
            clothes_id = clothes2label[clothes]
            pid2clothes[pid2label[pid], clothes_id] = 1
            if relabel:
                pid = pid2label[pid]
            else:
                pid = int(pid)

            # print("====================")
            # print(tracklet_path)
            camid = tracklet_path.split('/')[-1]

            print(camid)

            # print("====================")
            # print(camid)
            # cam = tracklet_path.split('_')[1]
            # if session == 'session3':
            #     camid = int(cam) + 12
            # else:
            #     camid = int(cam)

            num_imgs_per_tracklet.append(len(img_paths))
            tracklets.append((img_paths, pid, camid, clothes_id))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_clothes, pid2clothes, clothes2label

    def _process_data3(self, data_path, relabel=False, clothes2label=None):
        tracklet_path_list = []
        pid_container = set()
        clothes_container = set()
        with open(data_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                tracklet_path_list.append((tracklet_path, pid, clothes_label))
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        if clothes2label is None:
            clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        num_tracklets = len(tracklet_path_list)
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        tracklets = []
        num_imgs_per_tracklet = []
        pid2clothes = np.zeros((num_pids, len(clothes2label)))

        for tracklet_path, pid, clothes_label in tracklet_path_list:
            img_paths = glob.glob(osp.join(self.root, tracklet_path, '*.jpg'))
            img_paths.sort()

            clothes = '{}_{}'.format(pid, clothes_label)
            clothes_id = clothes2label[clothes]
            pid2clothes[pid2label[pid], clothes_id] = 1
            if relabel:
                pid = pid2label[pid]
            else:
                pid = int(pid)

            camid = tracklet_path.split('/')[-1]

            camid = int(camid)

            num_imgs_per_tracklet.append(len(img_paths))
            tracklets.append((img_paths, pid, camid, clothes_id))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_clothes, pid2clothes, clothes2label

    def _densesampling_for_trainingset(self, dataset, sampling_step=64):
        ''' Split all videos in training set into lots of clips for dense sampling.

        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
            sampling_step (int): sampling step for dense sampling

        Returns:
            new_dataset (list): output dataset
        '''
        new_dataset = []
        for (img_paths, pid, camid, clothes_id) in dataset:
            if sampling_step != 0:
                num_sampling = len(img_paths) // sampling_step
                if num_sampling == 0:
                    new_dataset.append((img_paths, pid, camid, clothes_id))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            new_dataset.append((img_paths[idx * sampling_step:], pid, camid, clothes_id))
                        else:
                            new_dataset.append(
                                (img_paths[idx * sampling_step: (idx + 1) * sampling_step], pid, camid, clothes_id))
            else:
                new_dataset.append((img_paths, pid, camid, clothes_id))

        return new_dataset

    def _recombination_for_testset(self, dataset, seq_len=16, stride=4):
        ''' Split all videos in test set into lots of equilong clips.

        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
            seq_len (int): sequence length of each output clip
            stride (int): temporal sampling stride

        Returns:
            new_dataset (list): output dataset with lots of equilong clips
            vid2clip_index (list): a list contains the start and end clip index of each original video
        '''
        new_dataset = []
        vid2clip_index = np.zeros((len(dataset), 2), dtype=int)
        for idx, (img_paths, pid, camid, clothes_id) in enumerate(dataset):
            # start index
            vid2clip_index[idx, 0] = len(new_dataset)
            # process the sequence that can be divisible by seq_len*stride
            for i in range(len(img_paths) // (seq_len * stride)):
                for j in range(stride):
                    begin_idx = i * (seq_len * stride) + j
                    end_idx = (i + 1) * (seq_len * stride)
                    clip_paths = img_paths[begin_idx: end_idx: stride]
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
            # process the remaining sequence that can't be divisible by seq_len*stride
            if len(img_paths) % (seq_len * stride) != 0:
                # reducing stride
                new_stride = (len(img_paths) % (seq_len * stride)) // seq_len
                for i in range(new_stride):
                    begin_idx = len(img_paths) // (seq_len * stride) * (seq_len * stride) + i
                    end_idx = len(img_paths) // (seq_len * stride) * (seq_len * stride) + seq_len * new_stride
                    clip_paths = img_paths[begin_idx: end_idx: new_stride]
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
                # process the remaining sequence that can't be divisible by seq_len
                if len(img_paths) % seq_len != 0:
                    clip_paths = img_paths[len(img_paths) // seq_len * seq_len:]
                    # loop padding
                    while len(clip_paths) < seq_len:
                        for index in clip_paths:
                            if len(clip_paths) >= seq_len:
                                break
                            clip_paths.append(index)
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
            # end index
            vid2clip_index[idx, 1] = len(new_dataset)
            assert ((vid2clip_index[idx, 1] - vid2clip_index[idx, 0]) == math.ceil(len(img_paths) / seq_len))

        return new_dataset, vid2clip_index.tolist()

class RVCCReID(object):
    """ V3DGait

    Reference:

    """

    def __init__(self, root='/data/datasets/', sampling_step=64, seq_len=16, stride=4, **kwargs):

        self.root = osp.join(root, 'RVCCReID')
        # print(self.root)
        self.train_path = osp.join(self.root, 'withTrain/train.txt')
        self.query_path = osp.join(self.root, 'withTrain/query.txt')
        self.gallery_path = osp.join(self.root, 'withTrain/gallery.txt')
        self._check_before_run()

        train, num_train_tracklets, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes, _ = \
            self._process_data3(self.train_path, relabel=True)
        clothes2label = self._clothes2label_test(self.query_path, self.gallery_path)
        query, num_query_tracklets, num_query_pids, num_query_imgs, num_query_clothes, _, _ = \
            self._process_data3(self.query_path, relabel=False, clothes2label=clothes2label)
        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs, num_gallery_clothes, _, _ = \
            self._process_data3(self.gallery_path, relabel=False, clothes2label=clothes2label)

        # slice each full-length video in the trainingset into more video clip
        train_dense = self._densesampling_for_trainingset(train, sampling_step)
        # In the test stage, each video sample is divided into a series of equilong video clips with a pre-defined stride.
        recombined_query, query_vid2clip_index = self._recombination_for_testset(query, seq_len=seq_len, stride=stride)
        recombined_gallery, gallery_vid2clip_index = self._recombination_for_testset(gallery, seq_len=seq_len,
                                                                                     stride=stride)

        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_clothes = num_train_clothes + len(clothes2label)
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

        # logger = logging.getLogger('reid.dataset')
        print("=> V3DGait loaded")
        print("Dataset statistics:")
        print("  ---------------------------------------------")
        print("  subset       | # ids | # tracklets | # clothes")
        print("  ---------------------------------------------")
        print(
            "  train        | {:5d} | {:11d} | {:9d}".format(num_train_pids, num_train_tracklets, num_train_clothes))
        print(
            "  train_dense  | {:5d} | {:11d} | {:9d}".format(num_train_pids, len(train_dense), num_train_clothes))
        print(
            "  query        | {:5d} | {:11d} | {:9d}".format(num_query_pids, num_query_tracklets, num_query_clothes))
        print("  gallery      | {:5d} | {:11d} | {:9d}".format(num_gallery_pids, num_gallery_tracklets,
                                                                     num_gallery_clothes))
        print("  ---------------------------------------------")
        print(
            "  total        | {:5d} | {:11d} | {:9d}".format(num_total_pids, num_total_tracklets, num_total_clothes))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ---------------------------------------------")

        # print(query_vid2clip_index)
        # print(len(query_vid2clip_index))

        self.train = train
        self.train_dense = train_dense
        self.query = query
        self.gallery = gallery

        self.recombined_query = recombined_query
        self.recombined_gallery = recombined_gallery
        self.query_vid2clip_index = query_vid2clip_index
        self.gallery_vid2clip_index = gallery_vid2clip_index

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_path):
            raise RuntimeError("'{}' is not available".format(self.train_path))
        if not osp.exists(self.query_path):
            raise RuntimeError("'{}' is not available".format(self.query_path))
        if not osp.exists(self.gallery_path):
            raise RuntimeError("'{}' is not available".format(self.gallery_path))

    def _clothes2label_test(self, query_path, gallery_path):
        pid_container = set()
        clothes_container = set()
        with open(query_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        with open(gallery_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        return clothes2label

    def _process_data(self, data_path, relabel=False, clothes2label=None):
        tracklet_path_list = []
        pid_container = set()
        clothes_container = set()
        with open(data_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                tracklet_path_list.append((tracklet_path, pid, clothes_label))
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        if clothes2label is None:
            clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        num_tracklets = len(tracklet_path_list)
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        tracklets = []
        num_imgs_per_tracklet = []
        pid2clothes = np.zeros((num_pids, len(clothes2label)))

        for tracklet_path, pid, clothes_label in tracklet_path_list:
            img_paths = glob.glob(osp.join(self.root, tracklet_path, '*'))
            img_paths.sort()

            clothes = '{}_{}'.format(pid, clothes_label)
            clothes_id = clothes2label[clothes]
            pid2clothes[pid2label[pid], clothes_id] = 1
            if relabel:
                pid = pid2label[pid]
            else:
                pid = int(pid)
            session = tracklet_path.split('/')[0]
            cam = tracklet_path.split('_')[1]
            if session == 'session3':
                camid = int(cam) + 12
            else:
                camid = int(cam)

            num_imgs_per_tracklet.append(len(img_paths))
            tracklets.append((img_paths, pid, camid, clothes_id))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_clothes, pid2clothes, clothes2label

    def _process_data2(self, data_path, relabel=False, clothes2label=None):
        tracklet_path_list = []
        pid_container = set()
        clothes_container = set()
        with open(data_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                tracklet_path_list.append((tracklet_path, pid, clothes_label))
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        if clothes2label is None:
            clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        num_tracklets = len(tracklet_path_list)
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        tracklets = []
        num_imgs_per_tracklet = []
        pid2clothes = np.zeros((num_pids, len(clothes2label)))

        for tracklet_path, pid, clothes_label in tracklet_path_list:
            img_paths = glob.glob(osp.join(self.root, tracklet_path, '*.png'))
            # print("+++++++++++++")
            # print(osp.join(self.root,tracklet_path,'*.jpg'))
            # print(img_paths)
            img_paths.sort()

            clothes = '{}_{}'.format(pid, clothes_label)
            clothes_id = clothes2label[clothes]
            pid2clothes[pid2label[pid], clothes_id] = 1
            if relabel:
                pid = pid2label[pid]
            else:
                pid = int(pid)

            # print("====================")
            # print(tracklet_path)
            camid = tracklet_path.split('/')[-1]

            print(camid)

            # print("====================")
            # print(camid)
            # cam = tracklet_path.split('_')[1]
            # if session == 'session3':
            #     camid = int(cam) + 12
            # else:
            #     camid = int(cam)

            num_imgs_per_tracklet.append(len(img_paths))
            tracklets.append((img_paths, pid, camid, clothes_id))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_clothes, pid2clothes, clothes2label

    def _process_data3(self, data_path, relabel=False, clothes2label=None):
        tracklet_path_list = []
        pid_container = set()
        clothes_container = set()
        with open(data_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                tracklet_path = 'image/'+tracklet_path

                # print(tracklet_path)

                tracklet_path_list.append((tracklet_path, pid, clothes_label))
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        if clothes2label is None:
            clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        num_tracklets = len(tracklet_path_list)
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        tracklets = []
        num_imgs_per_tracklet = []
        pid2clothes = np.zeros((num_pids, len(clothes2label)))

        for tracklet_path, pid, clothes_label in tracklet_path_list:
            img_paths = glob.glob(osp.join(self.root, tracklet_path, '*.jpg'))
            img_paths.sort()

            clothes = '{}_{}'.format(pid, clothes_label)
            clothes_id = clothes2label[clothes]
            pid2clothes[pid2label[pid], clothes_id] = 1
            if relabel:
                pid = pid2label[pid]
            else:
                pid = int(pid)
            # print(tracklet_path.split('/'))
            camid_1 = tracklet_path.split('/')[-2]
            # camid_2 = tracklet_path.split('/')[-1]
            #
            # camid = int(camid_1)*100 + int(camid_2)

            camid = int(camid_1)

            # print(camid)

            num_imgs_per_tracklet.append(len(img_paths))
            tracklets.append((img_paths, pid, camid, clothes_id))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_clothes, pid2clothes, clothes2label

    def _densesampling_for_trainingset(self, dataset, sampling_step=64):
        ''' Split all videos in training set into lots of clips for dense sampling.

        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
            sampling_step (int): sampling step for dense sampling

        Returns:
            new_dataset (list): output dataset
        '''
        new_dataset = []
        for (img_paths, pid, camid, clothes_id) in dataset:
            if sampling_step != 0:
                num_sampling = len(img_paths) // sampling_step
                if num_sampling == 0:
                    new_dataset.append((img_paths, pid, camid, clothes_id))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            new_dataset.append((img_paths[idx * sampling_step:], pid, camid, clothes_id))
                        else:
                            new_dataset.append(
                                (img_paths[idx * sampling_step: (idx + 1) * sampling_step], pid, camid, clothes_id))
            else:
                new_dataset.append((img_paths, pid, camid, clothes_id))

        return new_dataset

    def _recombination_for_testset(self, dataset, seq_len=16, stride=4):
        ''' Split all videos in test set into lots of equilong clips.

        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
            seq_len (int): sequence length of each output clip
            stride (int): temporal sampling stride

        Returns:
            new_dataset (list): output dataset with lots of equilong clips
            vid2clip_index (list): a list contains the start and end clip index of each original video
        '''
        new_dataset = []
        vid2clip_index = np.zeros((len(dataset), 2), dtype=int)
        for idx, (img_paths, pid, camid, clothes_id) in enumerate(dataset):
            # start index
            vid2clip_index[idx, 0] = len(new_dataset)
            # process the sequence that can be divisible by seq_len*stride
            for i in range(len(img_paths) // (seq_len * stride)):
                for j in range(stride):
                    begin_idx = i * (seq_len * stride) + j
                    end_idx = (i + 1) * (seq_len * stride)
                    clip_paths = img_paths[begin_idx: end_idx: stride]
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
            # process the remaining sequence that can't be divisible by seq_len*stride
            if len(img_paths) % (seq_len * stride) != 0:
                # reducing stride
                new_stride = (len(img_paths) % (seq_len * stride)) // seq_len
                for i in range(new_stride):
                    begin_idx = len(img_paths) // (seq_len * stride) * (seq_len * stride) + i
                    end_idx = len(img_paths) // (seq_len * stride) * (seq_len * stride) + seq_len * new_stride
                    clip_paths = img_paths[begin_idx: end_idx: new_stride]
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
                # process the remaining sequence that can't be divisible by seq_len
                if len(img_paths) % seq_len != 0:
                    clip_paths = img_paths[len(img_paths) // seq_len * seq_len:]
                    # loop padding
                    while len(clip_paths) < seq_len:
                        for index in clip_paths:
                            if len(clip_paths) >= seq_len:
                                break
                            clip_paths.append(index)
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
            # end index
            vid2clip_index[idx, 1] = len(new_dataset)
            assert ((vid2clip_index[idx, 1] - vid2clip_index[idx, 0]) == math.ceil(len(img_paths) / seq_len))

        return new_dataset, vid2clip_index.tolist()

# class RVCCReID_test(object):
#     """ V3DGait
#
#     Reference:
#
#     """
#
#     def __init__(self, root='/data/datasets/', sampling_step=64, seq_len=16, stride=4, **kwargs):
#
#         self.root = osp.join(root, 'RVCCReID')
#         # print(self.root)
#         self.train_path = osp.join(self.root, 'query.txt')
#         self.query_path = osp.join(self.root, 'query.txt')
#         self.gallery_path = osp.join(self.root, 'gallery.txt')
#         self._check_before_run()
#
#         train, num_train_tracklets, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes, _ = \
#             self._process_data3(self.train_path, relabel=True)
#         clothes2label = self._clothes2label_test(self.query_path, self.gallery_path)
#         query, num_query_tracklets, num_query_pids, num_query_imgs, num_query_clothes, _, _ = \
#             self._process_data3(self.query_path, relabel=False, clothes2label=clothes2label)
#         gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs, num_gallery_clothes, _, _ = \
#             self._process_data3(self.gallery_path, relabel=False, clothes2label=clothes2label)
#
#         # slice each full-length video in the trainingset into more video clip
#         train_dense = self._densesampling_for_trainingset(train, sampling_step)
#         # In the test stage, each video sample is divided into a series of equilong video clips with a pre-defined stride.
#         recombined_query, query_vid2clip_index = self._recombination_for_testset(query, seq_len=seq_len, stride=stride)
#         recombined_gallery, gallery_vid2clip_index = self._recombination_for_testset(gallery, seq_len=seq_len,
#                                                                                      stride=stride)
#
#         num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
#         min_num = np.min(num_imgs_per_tracklet)
#         max_num = np.max(num_imgs_per_tracklet)
#         avg_num = np.mean(num_imgs_per_tracklet)
#
#         num_total_pids = num_train_pids + num_gallery_pids
#         num_total_clothes = num_train_clothes + len(clothes2label)
#         num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets
#
#         # logger = logging.getLogger('reid.dataset')
#         print("=> V3DGait loaded")
#         print("Dataset statistics:")
#         print("  ---------------------------------------------")
#         print("  subset       | # ids | # tracklets | # clothes")
#         print("  ---------------------------------------------")
#         print(
#             "  train        | {:5d} | {:11d} | {:9d}".format(num_train_pids, num_train_tracklets, num_train_clothes))
#         print(
#             "  train_dense  | {:5d} | {:11d} | {:9d}".format(num_train_pids, len(train_dense), num_train_clothes))
#         print(
#             "  query        | {:5d} | {:11d} | {:9d}".format(num_query_pids, num_query_tracklets, num_query_clothes))
#         print("  gallery      | {:5d} | {:11d} | {:9d}".format(num_gallery_pids, num_gallery_tracklets,
#                                                                      num_gallery_clothes))
#         print("  ---------------------------------------------")
#         print(
#             "  total        | {:5d} | {:11d} | {:9d}".format(num_total_pids, num_total_tracklets, num_total_clothes))
#         print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
#         print("  ---------------------------------------------")
#
#         # print(query_vid2clip_index)
#         # print(len(query_vid2clip_index))
#
#         self.train = train
#         self.train_dense = train_dense
#         self.query = query
#         self.gallery = gallery
#
#         self.recombined_query = recombined_query
#         self.recombined_gallery = recombined_gallery
#         self.query_vid2clip_index = query_vid2clip_index
#         self.gallery_vid2clip_index = gallery_vid2clip_index
#
#         self.num_train_pids = num_train_pids
#         self.num_train_clothes = num_train_clothes
#         self.pid2clothes = pid2clothes
#
#     def _check_before_run(self):
#         """Check if all files are available before going deeper"""
#         if not osp.exists(self.root):
#             raise RuntimeError("'{}' is not available".format(self.root))
#         if not osp.exists(self.train_path):
#             raise RuntimeError("'{}' is not available".format(self.train_path))
#         if not osp.exists(self.query_path):
#             raise RuntimeError("'{}' is not available".format(self.query_path))
#         if not osp.exists(self.gallery_path):
#             raise RuntimeError("'{}' is not available".format(self.gallery_path))
#
#     def _clothes2label_test(self, query_path, gallery_path):
#         pid_container = set()
#         clothes_container = set()
#         with open(query_path, 'r') as f:
#             for line in f:
#                 new_line = line.rstrip()
#                 tracklet_path, pid, clothes_label = new_line.split()
#                 clothes = '{}_{}'.format(pid, clothes_label)
#                 pid_container.add(pid)
#                 clothes_container.add(clothes)
#         with open(gallery_path, 'r') as f:
#             for line in f:
#                 new_line = line.rstrip()
#                 tracklet_path, pid, clothes_label = new_line.split()
#                 clothes = '{}_{}'.format(pid, clothes_label)
#                 pid_container.add(pid)
#                 clothes_container.add(clothes)
#         pid_container = sorted(pid_container)
#         clothes_container = sorted(clothes_container)
#         pid2label = {pid: label for label, pid in enumerate(pid_container)}
#         clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}
#
#         return clothes2label
#
#     def _process_data(self, data_path, relabel=False, clothes2label=None):
#         tracklet_path_list = []
#         pid_container = set()
#         clothes_container = set()
#         with open(data_path, 'r') as f:
#             for line in f:
#                 new_line = line.rstrip()
#                 tracklet_path, pid, clothes_label = new_line.split()
#                 tracklet_path_list.append((tracklet_path, pid, clothes_label))
#                 clothes = '{}_{}'.format(pid, clothes_label)
#                 pid_container.add(pid)
#                 clothes_container.add(clothes)
#         pid_container = sorted(pid_container)
#         clothes_container = sorted(clothes_container)
#         pid2label = {pid: label for label, pid in enumerate(pid_container)}
#         if clothes2label is None:
#             clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}
#
#         num_tracklets = len(tracklet_path_list)
#         num_pids = len(pid_container)
#         num_clothes = len(clothes_container)
#
#         tracklets = []
#         num_imgs_per_tracklet = []
#         pid2clothes = np.zeros((num_pids, len(clothes2label)))
#
#         for tracklet_path, pid, clothes_label in tracklet_path_list:
#             img_paths = glob.glob(osp.join(self.root, tracklet_path, '*'))
#             img_paths.sort()
#
#             clothes = '{}_{}'.format(pid, clothes_label)
#             clothes_id = clothes2label[clothes]
#             pid2clothes[pid2label[pid], clothes_id] = 1
#             if relabel:
#                 pid = pid2label[pid]
#             else:
#                 pid = int(pid)
#             session = tracklet_path.split('/')[0]
#             cam = tracklet_path.split('_')[1]
#             if session == 'session3':
#                 camid = int(cam) + 12
#             else:
#                 camid = int(cam)
#
#             num_imgs_per_tracklet.append(len(img_paths))
#             tracklets.append((img_paths, pid, camid, clothes_id))
#
#         num_tracklets = len(tracklets)
#
#         return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_clothes, pid2clothes, clothes2label
#
#     def _process_data2(self, data_path, relabel=False, clothes2label=None):
#         tracklet_path_list = []
#         pid_container = set()
#         clothes_container = set()
#         with open(data_path, 'r') as f:
#             for line in f:
#                 new_line = line.rstrip()
#                 tracklet_path, pid, clothes_label = new_line.split()
#                 tracklet_path_list.append((tracklet_path, pid, clothes_label))
#                 clothes = '{}_{}'.format(pid, clothes_label)
#                 pid_container.add(pid)
#                 clothes_container.add(clothes)
#         pid_container = sorted(pid_container)
#         clothes_container = sorted(clothes_container)
#         pid2label = {pid: label for label, pid in enumerate(pid_container)}
#         if clothes2label is None:
#             clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}
#
#         num_tracklets = len(tracklet_path_list)
#         num_pids = len(pid_container)
#         num_clothes = len(clothes_container)
#
#         tracklets = []
#         num_imgs_per_tracklet = []
#         pid2clothes = np.zeros((num_pids, len(clothes2label)))
#
#         for tracklet_path, pid, clothes_label in tracklet_path_list:
#             img_paths = glob.glob(osp.join(self.root, tracklet_path, '*.png'))
#             # print("+++++++++++++")
#             # print(osp.join(self.root,tracklet_path,'*.jpg'))
#             # print(img_paths)
#             img_paths.sort()
#
#             clothes = '{}_{}'.format(pid, clothes_label)
#             clothes_id = clothes2label[clothes]
#             pid2clothes[pid2label[pid], clothes_id] = 1
#             if relabel:
#                 pid = pid2label[pid]
#             else:
#                 pid = int(pid)
#
#             # print("====================")
#             # print(tracklet_path)
#             camid = tracklet_path.split('/')[-1]
#
#             print(camid)
#
#             # print("====================")
#             # print(camid)
#             # cam = tracklet_path.split('_')[1]
#             # if session == 'session3':
#             #     camid = int(cam) + 12
#             # else:
#             #     camid = int(cam)
#
#             num_imgs_per_tracklet.append(len(img_paths))
#             tracklets.append((img_paths, pid, camid, clothes_id))
#
#         num_tracklets = len(tracklets)
#
#         return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_clothes, pid2clothes, clothes2label
#
#     def _process_data3(self, data_path, relabel=False, clothes2label=None):
#         tracklet_path_list = []
#         pid_container = set()
#         clothes_container = set()
#         with open(data_path, 'r') as f:
#             for line in f:
#                 new_line = line.rstrip()
#                 tracklet_path, pid, clothes_label = new_line.split()
#                 tracklet_path = 'image/'+tracklet_path
#                 tracklet_path_list.append((tracklet_path, pid, clothes_label))
#                 clothes = '{}_{}'.format(pid, clothes_label)
#                 pid_container.add(pid)
#                 clothes_container.add(clothes)
#         pid_container = sorted(pid_container)
#         clothes_container = sorted(clothes_container)
#         pid2label = {pid: label for label, pid in enumerate(pid_container)}
#         if clothes2label is None:
#             clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}
#
#         num_tracklets = len(tracklet_path_list)
#         num_pids = len(pid_container)
#         num_clothes = len(clothes_container)
#
#         tracklets = []
#         num_imgs_per_tracklet = []
#         pid2clothes = np.zeros((num_pids, len(clothes2label)))
#
#         for tracklet_path, pid, clothes_label in tracklet_path_list:
#             img_paths = glob.glob(osp.join(self.root, tracklet_path, '*.jpg'))
#             img_paths.sort()
#
#             clothes = '{}_{}'.format(pid, clothes_label)
#             clothes_id = clothes2label[clothes]
#             pid2clothes[pid2label[pid], clothes_id] = 1
#             if relabel:
#                 pid = pid2label[pid]
#             else:
#                 pid = int(pid)
#             # print(tracklet_path.split('/'))
#             camid_1 = tracklet_path.split('/')[-2]
#             # camid_2 = tracklet_path.split('/')[-1]
#             #
#             # camid = int(camid_1)*100 + int(camid_2)
#
#             camid = int(camid_1)
#
#             # print(camid)
#
#             num_imgs_per_tracklet.append(len(img_paths))
#             tracklets.append((img_paths, pid, camid, clothes_id))
#
#         num_tracklets = len(tracklets)
#
#         return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_clothes, pid2clothes, clothes2label
#
#     def _densesampling_for_trainingset(self, dataset, sampling_step=64):
#         ''' Split all videos in training set into lots of clips for dense sampling.
#
#         Args:
#             dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
#             sampling_step (int): sampling step for dense sampling
#
#         Returns:
#             new_dataset (list): output dataset
#         '''
#         new_dataset = []
#         for (img_paths, pid, camid, clothes_id) in dataset:
#             if sampling_step != 0:
#                 num_sampling = len(img_paths) // sampling_step
#                 if num_sampling == 0:
#                     new_dataset.append((img_paths, pid, camid, clothes_id))
#                 else:
#                     for idx in range(num_sampling):
#                         if idx == num_sampling - 1:
#                             new_dataset.append((img_paths[idx * sampling_step:], pid, camid, clothes_id))
#                         else:
#                             new_dataset.append(
#                                 (img_paths[idx * sampling_step: (idx + 1) * sampling_step], pid, camid, clothes_id))
#             else:
#                 new_dataset.append((img_paths, pid, camid, clothes_id))
#
#         return new_dataset
#
#     def _recombination_for_testset(self, dataset, seq_len=16, stride=4):
#         ''' Split all videos in test set into lots of equilong clips.
#
#         Args:
#             dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
#             seq_len (int): sequence length of each output clip
#             stride (int): temporal sampling stride
#
#         Returns:
#             new_dataset (list): output dataset with lots of equilong clips
#             vid2clip_index (list): a list contains the start and end clip index of each original video
#         '''
#         new_dataset = []
#         vid2clip_index = np.zeros((len(dataset), 2), dtype=int)
#         for idx, (img_paths, pid, camid, clothes_id) in enumerate(dataset):
#             # start index
#             vid2clip_index[idx, 0] = len(new_dataset)
#             # process the sequence that can be divisible by seq_len*stride
#             for i in range(len(img_paths) // (seq_len * stride)):
#                 for j in range(stride):
#                     begin_idx = i * (seq_len * stride) + j
#                     end_idx = (i + 1) * (seq_len * stride)
#                     clip_paths = img_paths[begin_idx: end_idx: stride]
#                     assert (len(clip_paths) == seq_len)
#                     new_dataset.append((clip_paths, pid, camid, clothes_id))
#             # process the remaining sequence that can't be divisible by seq_len*stride
#             if len(img_paths) % (seq_len * stride) != 0:
#                 # reducing stride
#                 new_stride = (len(img_paths) % (seq_len * stride)) // seq_len
#                 for i in range(new_stride):
#                     begin_idx = len(img_paths) // (seq_len * stride) * (seq_len * stride) + i
#                     end_idx = len(img_paths) // (seq_len * stride) * (seq_len * stride) + seq_len * new_stride
#                     clip_paths = img_paths[begin_idx: end_idx: new_stride]
#                     assert (len(clip_paths) == seq_len)
#                     new_dataset.append((clip_paths, pid, camid, clothes_id))
#                 # process the remaining sequence that can't be divisible by seq_len
#                 if len(img_paths) % seq_len != 0:
#                     clip_paths = img_paths[len(img_paths) // seq_len * seq_len:]
#                     # loop padding
#                     while len(clip_paths) < seq_len:
#                         for index in clip_paths:
#                             if len(clip_paths) >= seq_len:
#                                 break
#                             clip_paths.append(index)
#                     assert (len(clip_paths) == seq_len)
#                     new_dataset.append((clip_paths, pid, camid, clothes_id))
#             # end index
#             vid2clip_index[idx, 1] = len(new_dataset)
#             assert ((vid2clip_index[idx, 1] - vid2clip_index[idx, 0]) == math.ceil(len(img_paths) / seq_len))
#
#         return new_dataset, vid2clip_index.tolist()


class CCVID(object):
    """ CCVID

    Reference:
        Gu et al. Clothes-Changing Person Re-identification with RGB Modality Only. In CVPR, 2022.
    """

    def __init__(self, root='/data/datasets/', sampling_step=64, seq_len=16, stride=4, **kwargs):
        self.root = osp.join(root, 'image')
        self.train_path = osp.join(self.root, 'train.txt')
        self.query_path = osp.join(self.root, 'query.txt')
        self.gallery_path = osp.join(self.root, 'gallery.txt')
        self._check_before_run()

        train, num_train_tracklets, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes, _ = \
            self._process_data(self.train_path, relabel=True)
        clothes2label = self._clothes2label_test(self.query_path, self.gallery_path)
        query, num_query_tracklets, num_query_pids, num_query_imgs, num_query_clothes, _, _ = \
            self._process_data(self.query_path, relabel=False, clothes2label=clothes2label)
        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs, num_gallery_clothes, _, _ = \
            self._process_data(self.gallery_path, relabel=False, clothes2label=clothes2label)

        # slice each full-length video in the trainingset into more video clip
        train_dense = self._densesampling_for_trainingset(train, sampling_step)
        # In the test stage, each video sample is divided into a series of equilong video clips with a pre-defined stride.
        recombined_query, query_vid2clip_index = self._recombination_for_testset(query, seq_len=seq_len, stride=stride)
        recombined_gallery, gallery_vid2clip_index = self._recombination_for_testset(gallery, seq_len=seq_len,
                                                                                     stride=stride)

        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_clothes = num_train_clothes + len(clothes2label)
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

        # logger = logging.getLogger('reid.dataset')
        print("=> CCVID loaded")
        print("Dataset statistics:")
        print("  ---------------------------------------------")
        print("  subset       | # ids | # tracklets | # clothes")
        print("  ---------------------------------------------")
        print(
            "  train        | {:5d} | {:11d} | {:9d}".format(num_train_pids, num_train_tracklets, num_train_clothes))
        print(
            "  train_dense  | {:5d} | {:11d} | {:9d}".format(num_train_pids, len(train_dense), num_train_clothes))
        print(
            "  query        | {:5d} | {:11d} | {:9d}".format(num_query_pids, num_query_tracklets, num_query_clothes))
        print("  gallery      | {:5d} | {:11d} | {:9d}".format(num_gallery_pids, num_gallery_tracklets,
                                                                     num_gallery_clothes))
        print("  ---------------------------------------------")
        print(
            "  total        | {:5d} | {:11d} | {:9d}".format(num_total_pids, num_total_tracklets, num_total_clothes))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ---------------------------------------------")

        self.train = train
        self.train_dense = train_dense
        self.query = query
        self.gallery = gallery

        self.recombined_query = recombined_query
        self.recombined_gallery = recombined_gallery
        self.query_vid2clip_index = query_vid2clip_index
        self.gallery_vid2clip_index = gallery_vid2clip_index

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_path):
            raise RuntimeError("'{}' is not available".format(self.train_path))
        if not osp.exists(self.query_path):
            raise RuntimeError("'{}' is not available".format(self.query_path))
        if not osp.exists(self.gallery_path):
            raise RuntimeError("'{}' is not available".format(self.gallery_path))

    def _clothes2label_test(self, query_path, gallery_path):
        pid_container = set()
        clothes_container = set()
        with open(query_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        with open(gallery_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        return clothes2label

    def _process_data(self, data_path, relabel=False, clothes2label=None):
        tracklet_path_list = []
        pid_container = set()
        clothes_container = set()
        with open(data_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                tracklet_path_list.append((tracklet_path, pid, clothes_label))
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        if clothes2label is None:
            clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        num_tracklets = len(tracklet_path_list)
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        tracklets = []
        num_imgs_per_tracklet = []
        pid2clothes = np.zeros((num_pids, len(clothes2label)))

        for tracklet_path, pid, clothes_label in tracklet_path_list:
            img_paths = glob.glob(osp.join(self.root, tracklet_path, '*'))
            img_paths.sort()

            clothes = '{}_{}'.format(pid, clothes_label)
            clothes_id = clothes2label[clothes]
            pid2clothes[pid2label[pid], clothes_id] = 1
            if relabel:
                pid = pid2label[pid]
            else:
                pid = int(pid)
            session = tracklet_path.split('/')[0]
            cam = tracklet_path.split('_')[1]
            if session == 'session3':
                camid = int(cam) + 12
            else:
                camid = int(cam)

            num_imgs_per_tracklet.append(len(img_paths))
            tracklets.append((img_paths, pid, camid, clothes_id))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_clothes, pid2clothes, clothes2label

    def _densesampling_for_trainingset(self, dataset, sampling_step=64):
        ''' Split all videos in training set into lots of clips for dense sampling.

        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
            sampling_step (int): sampling step for dense sampling

        Returns:
            new_dataset (list): output dataset
        '''
        new_dataset = []
        for (img_paths, pid, camid, clothes_id) in dataset:
            if sampling_step != 0:
                num_sampling = len(img_paths) // sampling_step
                if num_sampling == 0:
                    new_dataset.append((img_paths, pid, camid, clothes_id))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            new_dataset.append((img_paths[idx * sampling_step:], pid, camid, clothes_id))
                        else:
                            new_dataset.append(
                                (img_paths[idx * sampling_step: (idx + 1) * sampling_step], pid, camid, clothes_id))
            else:
                new_dataset.append((img_paths, pid, camid, clothes_id))

        return new_dataset

    def _recombination_for_testset(self, dataset, seq_len=16, stride=4):
        ''' Split all videos in test set into lots of equilong clips.

        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
            seq_len (int): sequence length of each output clip
            stride (int): temporal sampling stride

        Returns:
            new_dataset (list): output dataset with lots of equilong clips
            vid2clip_index (list): a list contains the start and end clip index of each original video
        '''
        new_dataset = []
        vid2clip_index = np.zeros((len(dataset), 2), dtype=int)
        for idx, (img_paths, pid, camid, clothes_id) in enumerate(dataset):
            # start index
            vid2clip_index[idx, 0] = len(new_dataset)
            # process the sequence that can be divisible by seq_len*stride
            for i in range(len(img_paths) // (seq_len * stride)):
                for j in range(stride):
                    begin_idx = i * (seq_len * stride) + j
                    end_idx = (i + 1) * (seq_len * stride)
                    clip_paths = img_paths[begin_idx: end_idx: stride]
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
            # process the remaining sequence that can't be divisible by seq_len*stride
            if len(img_paths) % (seq_len * stride) != 0:
                # reducing stride
                new_stride = (len(img_paths) % (seq_len * stride)) // seq_len
                for i in range(new_stride):
                    begin_idx = len(img_paths) // (seq_len * stride) * (seq_len * stride) + i
                    end_idx = len(img_paths) // (seq_len * stride) * (seq_len * stride) + seq_len * new_stride
                    clip_paths = img_paths[begin_idx: end_idx: new_stride]
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
                # process the remaining sequence that can't be divisible by seq_len
                if len(img_paths) % seq_len != 0:
                    clip_paths = img_paths[len(img_paths) // seq_len * seq_len:]
                    # loop padding
                    while len(clip_paths) < seq_len:
                        for index in clip_paths:
                            if len(clip_paths) >= seq_len:
                                break
                            clip_paths.append(index)
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
            # end index
            vid2clip_index[idx, 1] = len(new_dataset)
            assert ((vid2clip_index[idx, 1] - vid2clip_index[idx, 0]) == math.ceil(len(img_paths) / seq_len))

        return new_dataset, vid2clip_index.tolist()


__factory = {
    'mars': Mars,
    'ilidsvid': iLIDSVID,
    'prid': PRID,
    'lsvid': LSVID,
    'duke': DukeMTMCVidReID,
    'vccvid':VCCVID,
    'v3dgait':V3DGait,
    'svreid_extend':V3DGait_extend,
    'ccvid':CCVID,
    'rvccvid':RVCCReID,
    'duke2':DukeMTMCreID,
}


def get_names():
    return __factory.keys()


def init_dataset(name, root=None, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))

    return __factory[name](root=root, *args, **kwargs)
    # return __factory[name]( *args, **kwargs)
