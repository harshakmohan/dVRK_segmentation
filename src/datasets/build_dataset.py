import os
import os.path as osp
import random
import torch
import numpy as np
import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import Dataset

'''
Grab data and create all the necessary dataloaders.

All data preprocessing should happen here.
'''
# TODO: @Harsha flesh out Datasets and Dataloader functions

class UCL(Dataset):
    def __init__(self, data_folder: str, select_all: bool, video_paths: list):
        '''

        :param data_folder: From argparse, "ucl_data_dir" which points to UCL dataset
        :param select_all: From argparse, if this is True, then load the entire UCL dataset.
        :param train_videos: From argparse, this is an int which tells which videos to use for training
        :param test_videos: From argparse, this is an int which tells which videos to use for testing
        '''
        self.data_folder = data_folder
        self.video_paths = None
        self.img_paths = []
        self.mask_paths = []

        if select_all:
            '''
            Open all the UCL videos and load them into the Dataset. 80/20 train/test split
            '''
            self.video_paths = [ file for file in os.listdir(data_folder) if file.startswith('Video')]
            sample = random.sample(self.video_paths, k=14)

            for video in sample:
                for i in range(300):
                    name = f'{str(i).zfill(3)}.png'
                    self.img_paths.append( osp.join(osp.join(osp.join(self.data_folder, video), 'images'), name) )
                    self.mask_paths.append( osp.join(osp.join(osp.join(self.data_folder, video), 'ground_truth'),name) )
        else:
            '''
            Use the list in video_paths and load those videos only.
            Example of video_paths input: ['Video_01', 'Video_02', 'Video_04', 'Video_05']
            '''
            self.video_paths = video_paths

            for video in self.video_paths:
                for i in range(300):
                    name = f'{str(i).zfill(3)}.png'
                    self.img_paths.append(osp.join(osp.join(osp.join(self.data_folder, video), 'images'), name))
                    self.mask_paths.append(osp.join(osp.join(osp.join(self.data_folder, video), 'ground_truth'), name))


    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, index: int):

        image = torch.from_numpy(np.array(Image.open(self.img_paths[index]).convert('RGB')) / 255.0).float()
        image = torch.permute(image, (2, 0, 1))
        mask = torch.from_numpy(np.array(Image.open(self.mask_paths[index]).convert('L'), dtype=np.float32) / 255.0).float()

        return image, mask


class EndoVis(Dataset):
    def __init__(self, data_folder: str):
        self.image_paths = []
        self.mask_paths = []

    def __len__(self):
        pass

    def __getitem__(self):
        image = 0
        mask = 0
        return image, mask


def build_dataloaders():
    pass

