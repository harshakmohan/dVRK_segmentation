import os
import torch
import numpy as np
import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import DataLoader
from src.datasets.build_dataset import UCL, BinaryEndoVis

def UCL_dataloader(data_dir, batch_size, train_videos, test_videos):

    train_ds = UCL(data_folder=data_dir, select_all=False, video_paths=train_videos, transform=True)
    test_ds = UCL(data_folder=data_dir, select_all=False, video_paths=test_videos, transform=False)

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

def Endovis_Binary_dataloader(data_dir, batch_size):

    binary_endo_dataset = BinaryEndoVis(data_dir)
    #test_ds = BinaryEndoVis(data_dir)

    test_size = int(0.1*len(binary_endo_dataset))
    train_size = len(binary_endo_dataset)-test_size

    train_ds, test_ds = torch.utils.data.random_split(binary_endo_dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

