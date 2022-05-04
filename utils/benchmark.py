from os import path as osp
import time
from src.datasets.build_dataloader import Endovis_Binary_dataloader
from src.models import UNET
from src.run.utils import dice, check_accuracy
import torch
import torchvision

folder = osp.abspath("../predictions/")
print(folder)
batch_size = 1
control_loader, _ = Endovis_Binary_dataloader(osp.abspath('../data/benchmarking/control'), batch_size)
x2_loader, _ = Endovis_Binary_dataloader(osp.abspath('../data/benchmarking/scaled_2'), batch_size)
x4_loader, _ = Endovis_Binary_dataloader(osp.abspath('../data/benchmarking/scaled_4'), batch_size)
x8_loader, _ = Endovis_Binary_dataloader(osp.abspath('../data/benchmarking/scaled_8'), batch_size)
x16_loader, _ = Endovis_Binary_dataloader(osp.abspath('../data/benchmarking/scaled_16'), batch_size)
x32_loader, _ = Endovis_Binary_dataloader(osp.abspath('../data/benchmarking/scaled_32'), batch_size)

model = UNET()
model.load_state_dict(torch.load(osp.abspath('../model_checkpoints/unet_ucl_epoch5_84ds.pth'), map_location='cpu'))

loaders = [x2_loader, x4_loader, x8_loader, control_loader, x16_loader, x32_loader]
log = []
for i, loader in enumerate(loaders):
    total_loader_time = 0
    print('Loader ', i)
    for batch_idx, (data, targets) in enumerate(loader):
        start = time.time()
        pred = torch.sigmoid(model(data))
        pred = (pred > .5).float()
        end = time.time()
        if batch_idx % 50 == 0:
            torchvision.utils.save_image(pred, f"{folder}/pred_{batch_idx}_loader{i}.png")
            torchvision.utils.save_image(targets, f"{folder}/{batch_idx}_loader{i}.png")
            print(f'Predictions saved at image {batch_idx}')
            print(f'average time elapsed is {total_loader_time/(batch_idx+1)}')
        elapsed = end-start
        total_loader_time += elapsed
        pred = pred.squeeze()
        targets = targets.squeeze()
    dice = check_accuracy(loader, model)
    log.append([total_loader_time/(batch_idx+1), dice])

print(data)
