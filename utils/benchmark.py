from os import path as osp
import time
from src.datasets.build_dataloader import Endovis_Binary_dataloader
from src.models import UNET
import torch
import tqdm

batch_size = 1
control_loader, _ = Endovis_Binary_dataloader(osp.abspath('../data/benchmarking/control'), batch_size)
x2_loader, _ = Endovis_Binary_dataloader(osp.abspath('../data/benchmarking/scaled_2'), batch_size)
x4_loader, _ = Endovis_Binary_dataloader(osp.abspath('../data/benchmarking/scaled_4'), batch_size)
x8_loader, _ = Endovis_Binary_dataloader(osp.abspath('../data/benchmarking/scaled_8'), batch_size)
x16_loader, _ = Endovis_Binary_dataloader(osp.abspath('../data/benchmarking/scaled_16'), batch_size)
x32_loader, _ = Endovis_Binary_dataloader(osp.abspath('../data/benchmarking/scaled_32'), batch_size)

model = UNET()
model.load_state_dict(torch.load(osp.abspath('../model_checkpoints/unet_ucl_epoch10_74ds.pth'), map_location='cpu'))

loaders = [control_loader, x2_loader, x4_loader, x8_loader, x16_loader, x32_loader]

for loader in loaders:
    for batch_idx, (data, targets) in enumerate(loader): 
        pred = torch.sigmoid(model(data))
        print(pred)