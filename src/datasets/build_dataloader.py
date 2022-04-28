import os
import torch
import numpy as np
import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import DataLoader
from src.datasets.build_dataset import UCL, BinaryEndoVis

def UCL_dataloader(data_dir, batch_size, train_videos, test_videos):

    train_ds = UCL(data_folder=data_dir, select_all=False, video_paths=train_videos)
    test_ds = UCL(data_folder=data_dir, select_all=False, video_paths=test_videos)

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

def Endovis_Binary_dataloader(data_dir, batch_size):

    train_ds = BinaryEndoVis(data_dir)
    #test_ds = BinaryEndoVis(data_dir)

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    #test_dataloader = None
    return train_dataloader#, test_dataloader

