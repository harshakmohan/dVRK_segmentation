from os import path as osp
from os import mkdir
from time import time, strftime
from datetime import datetime
from src.datasets.build_dataloader import Endovis_Binary_dataloader
from src.models import UNET
from src.run.utils import dice, check_accuracy
import torch
import torchvision
import pandas as pd

run = datetime.now().strftime("%j%H%S")
folder = osp.abspath(f"../predictions/{run}")
if not osp.exists(folder):
    mkdir(folder)

batch_size = 1
control_loader, _ = Endovis_Binary_dataloader(osp.abspath('../data/benchmarking/control'), batch_size)
x50_loader, _ = Endovis_Binary_dataloader(osp.abspath('../data/benchmarking/scaled_50'), batch_size)
x25_loader, _ = Endovis_Binary_dataloader(osp.abspath('../data/benchmarking/scaled_25'), batch_size)
x12_loader, _ = Endovis_Binary_dataloader(osp.abspath('../data/benchmarking/scaled_12'), batch_size)
x6_loader, _ = Endovis_Binary_dataloader(osp.abspath('../data/benchmarking/scaled_6'), batch_size)

model = UNET()
model.load_state_dict(torch.load(osp.abspath('../model_checkpoints/unet_ucl_epoch5_84ds.pth'), map_location='cpu'))

loaders = {
    50: x50_loader, 
    25: x25_loader, 
    12: x12_loader, 
    6: x6_loader, 
    100: control_loader
}
log = []

for k, loader in loaders.items():
    total_loader_time = 0
    print('Loader ', k)
    for batch_idx, (data, targets) in enumerate(loader):
        start = time()
        pred = torch.sigmoid(model(data))
        pred = (pred > .5).float()
        end = time()
        if batch_idx % 50 == 0:
            torchvision.utils.save_image(pred, f"{folder}/scale{k}_{batch_idx}_pred.png")
            torchvision.utils.save_image(targets, f"{folder}/scale{k}_{batch_idx}_img.png")
            print(f'Predictions saved at image {batch_idx}')
            print(f'average time elapsed is {total_loader_time/(batch_idx+1)}')
        elapsed = end-start
        total_loader_time += elapsed
        pred = pred.squeeze()
        targets = targets.squeeze()
    dice = check_accuracy(loader, model)
    log.append([k, total_loader_time/(batch_idx+1), dice])

df = pd.DataFrame(log, columns=['Scaling Factor', 'Avg Time', 'Dice Score'])
df.to_csv(f'{folder}/log.csv')
