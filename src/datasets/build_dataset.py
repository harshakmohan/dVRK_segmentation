import os
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
    def __init__(self):
        pass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):

        image = torch.from_numpy(np.array(Image.open(self.image_paths[index]).convert('RGB')) / 255.0).float()
        image = torch.permute(image, (2, 0, 1))
        mask = torch.from_numpy(np.array(Image.open(self.mask_paths[index]).convert('L'), dtype=np.float32) / 255.0).float()

        return image, mask


class EndoVis(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass

def build_dataloaders():
    pass

