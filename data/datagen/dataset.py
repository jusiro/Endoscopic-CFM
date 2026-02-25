
import torch
import glob
import os
import random

from torch.utils.data import Dataset as _TorchDataset

from data.datagen.utils import read_img_seq

# Base path for datasets and outputs.
from docs.local_data.constants import PATH_DATASETS
from docs.local_data.constants import PATH_EXPERIMENTS

class SRDataset(_TorchDataset):

    def __init__(self, opt, split=None):

        self.data = []

        # Set subfolders path for low and high resolution.
        if split is None:
            subfolders_lq = sorted(glob.glob(os.path.join(PATH_DATASETS, opt['dataset']["relpath"], opt['relpath_input'], '*')))
            subfolders_gt = sorted(glob.glob(os.path.join(PATH_DATASETS, opt['dataset']["relpath"], opt['relpath_gt'], '*')))
        else:
            with open(split, 'r') as fin:
                subfolders = [line.split('\n')[0].replace(" ", "") for line in fin]
            subfolders_lq = sorted([glob.glob(os.path.join(PATH_DATASETS, opt['dataset']["relpath"] + '*', opt['dataset']['relpath_input'], '*' + key + '*'))[0] for key in subfolders])
            subfolders_gt = sorted([glob.glob(os.path.join(PATH_DATASETS, opt['dataset']["relpath"] + '*', opt['dataset']['relpath_gt'], '*' + key + '*'))[0] for key in subfolders])

        # Remove non-folder entries
        subfolders_lq = [i for i in subfolders_lq if "." not in i.split("/")[-1]]
        subfolders_gt = [i for i in subfolders_gt if "." not in i.split("/")[-1]]

        # Read images for each subfolder
        for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
            # Read image names.
            img_lq = os.listdir(subfolder_lq)
            img_gt = os.listdir(subfolder_gt)

            # Filter non-desired files.
            img_lq = set(img_lq) - set(opt['dataset']['ignore_files'])
            img_gt = set(img_gt) - set(opt['dataset']['ignore_files'])

            # Sort files as sequence.
            img_lq = sorted(img_lq, key=lambda x: int(x.split("_")[-1].split('.')[0]))
            img_gt = sorted(img_gt, key=lambda x: int(x.split("_")[-1].split('.')[0]))

            # Add absolute path
            img_paths_lq = [os.path.join(subfolder_lq, ifile) for ifile in img_lq]
            img_paths_gt = [os.path.join(subfolder_gt, ifile) for ifile in img_gt]

            # Add entry to data
            self.data.append({'img_paths_lq': img_paths_lq,
                              'img_paths_gt': img_paths_gt,
                              'img_names': img_gt,
                              'folder': [subfolder_gt.split('/')[-1]]
                              })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Read video sequences
        imgs_lq = read_img_seq(self.data[index]['img_paths_lq'])
        imgs_gt = read_img_seq(self.data[index]['img_paths_gt'])

        # Create out dict
        item = {
            'imgs_lq': imgs_lq,  # (t, c, h, w)
            'imgs_gt': imgs_gt,  # (t, c, h, w)
            'img_names': self.data[index]['img_names'],
            'folder': self.data[index]['folder']
        }

        return item


class ErrorDataset(_TorchDataset):

    def __init__(self, opt, split=None):

        self.data = []

        # Set subfolders path for low and high resolution.
        if split is None:
            subfolders_feat = sorted(glob.glob(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Features', '*')))
            subfolders_gt = sorted(glob.glob(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Error', '*')))
        else:
            with open(split, 'r') as fin:
                subfolders = [line.split('\n')[0].replace(" ", "") for line in fin]
            # Subsample training data for ablation studies
            if 'train' in split:
                if 'ratio_train' in opt['errornet_train'].keys():
                    subfolders = random.sample(subfolders, int(len(subfolders) * opt['errornet_train']['ratio_train']))
            subfolders_feat = sorted([glob.glob(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Features', '*' + key + '*'))[0] for key in subfolders])
            subfolders_gt = sorted([glob.glob(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Error', '*' + key + '*'))[0] for key in subfolders])

        # Remove non-folder entries
        subfolders_feat = [i for i in subfolders_feat if "." not in i.split("/")[-1]]
        subfolders_gt = [i for i in subfolders_gt if "." not in i.split("/")[-1]]

        # Read images for each subfolder
        for subfolder_feat, subfolder_gt in zip(subfolders_feat, subfolders_gt):
            # Read image names.
            feats = os.listdir(subfolder_feat)
            img_gt = os.listdir(subfolder_gt)

            # Sort files as sequence.
            feats = sorted(feats, key=lambda x: int(x.split("_")[-1].split('.')[0]))
            img_gt = sorted(img_gt, key=lambda x: int(x.split("_")[-1].split('.')[0]))

            # Add absolute path
            feats_paths = [os.path.join(subfolder_feat, ifile) for ifile in feats]
            gt_paths = [os.path.join(subfolder_gt, ifile) for ifile in img_gt]

            # Add entry to data
            self.data.append({'feats_paths': feats_paths,
                              'gt_paths': gt_paths,
                              'img_names': img_gt,
                              'folder': [subfolder_gt.split('/')[-1]]
                              })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Read video sequences
        feats = torch.stack([torch.load(ifile) for ifile in self.data[index]['feats_paths']], dim=1).squeeze()
        gt = torch.stack([torch.load(ifile) for ifile in self.data[index]['gt_paths']], dim=0).squeeze()

        # Create out dict
        item = {
            'feats': feats,  # (t, c, h, w)
            'gt': gt,  # (t, c, h, w)
            'img_names': self.data[index]['img_names'],
            'folder': self.data[index]['folder']
        }

        return item


class CRCDataset(_TorchDataset):

    def __init__(self, opt, split=None):

        self.data = []

        # Set subfolders path for low and high resolution.
        if split is None:
            subfolders_prederror = sorted(glob.glob(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Errornet-' + opt['errornet']['subconfig'], 'PredError', '*')))
            subfolders_gt = sorted(glob.glob(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Error', '*')))
        else:
            with open(split, 'r') as fin:
                subfolders = [line.split('\n')[0].replace(" ", "") for line in fin]
            subfolders_prederror = sorted([glob.glob(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Errornet-' + opt['errornet']['subconfig'], 'PredError', '*' + key + '*'))[0] for key in subfolders])
            subfolders_gt = sorted([glob.glob(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Error', '*' + key + '*'))[0] for key in subfolders])

        # Remove non-folder entries
        subfolders_prederror = [i for i in subfolders_prederror if "." not in i.split("/")[-1]]
        subfolders_gt = [i for i in subfolders_gt if "." not in i.split("/")[-1]]

        # Read images for each subfolder
        for subfolder_prederror, subfolder_gt in zip(subfolders_prederror, subfolders_gt):
            # Read image names.
            prederror = os.listdir(subfolder_prederror)
            img_gt = os.listdir(subfolder_gt)

            # Sort files as sequence.
            prederror = sorted(prederror, key=lambda x: int(x.split("_")[-1].split('.')[0]))
            img_gt = sorted(img_gt, key=lambda x: int(x.split("_")[-1].split('.')[0]))

            # Add absolute path
            prederror_paths = [os.path.join(subfolder_prederror, ifile) for ifile in prederror]
            gt_paths = [os.path.join(subfolder_gt, ifile) for ifile in img_gt]

            # Add entry to data
            self.data.append({'prederror_paths': prederror_paths,
                              'gt_paths': gt_paths,
                              'img_names': img_gt,
                              'folder': [subfolder_gt.split('/')[-1]]
                              })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Read video sequences
        prederror = torch.stack([torch.load(ifile) for ifile in self.data[index]['prederror_paths']], dim=0).squeeze()
        gt = torch.stack([torch.load(ifile) for ifile in self.data[index]['gt_paths']], dim=0).squeeze()

        # Create out dict
        item = {
            'prederror': prederror,  # (t, c, h, w)
            'gt': gt,  # (t, c, h, w)
            'img_names': self.data[index]['img_names'],
            'folder': self.data[index]['folder']
        }

        return item