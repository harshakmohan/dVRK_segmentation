import os
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
    def __init__(self, data_folder: str, random_selection: bool, train_videos: int, test_videos: int):
        '''

        :param data_folder: From argparse, "ucl_data_dir" which points to UCL dataset
        :param random_selection: From argparse, if this is True, then let's do an 80/20 train/test split for the entire UCL dataset.
        :param train_videos: From argparse, this is an int which tells which videos to use for training
        :param test_videos: From argparse, this is an int which tells which videos to use for testing
        '''
        self.data_folder = data_folder
        self.video_paths = None
        self.img_paths = []
        self.mask_paths = []

        if random_selection:
            '''
            Open all the UCL videos and load them into the Dataset. 80/20 train/test split
            '''
            self.video_paths = [ file for file in os.listdir(data_folder) if file.startswith('Video')]
            sample = random.sample(self.video_paths, k=14)

            for video in sample:
                vpath_img = os.path.join(video, 'images')
                vpath_mask = os.path.join(video, 'ground_truth')

                self.img_paths.append(
                    os.path.join(self.data_folder, vpath_img))
                self.mask_paths.append(
                    os.path.join(self.data_folder, vpath_mask))

        else:
            '''
            Use the int vals in train_videos and test_videos to assign videos to train and test.
            '''

        print(len(self.img_paths))
        print(len(self.mask_paths))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        pass
        #
        # image = torch.from_numpy(np.array(Image.open(self.img_paths[index]).convert('RGB')) / 255.0).float()
        # image = torch.permute(image, (2, 0, 1))
        # mask = torch.from_numpy(np.array(Image.open(self.mask_paths[index]).convert('L'), dtype=np.float32) / 255.0).float()
        #
        # return image, mask


class EndoVis(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass


def build_dataloaders():
    pass

