'''
Define train loop here
'''

# TODO: @Harsha Flesh out train loop function

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from .src.models import UNET
from .utils import (load_checkpoint, save_checkpoint, check_accuracy, save_predictions_as_imgs)
from .utils import  DiceLoss2D
import os

def train_fn(loader, model, optimizer, loss_fn, device):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            print("loss = ", loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())