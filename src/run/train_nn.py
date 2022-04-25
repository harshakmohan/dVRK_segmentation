'''
Train Loop
'''

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from src.models import UNET
from .utils import ( save_checkpoint, check_accuracy )
from .utils import DiceLoss2D
import os

def train_fn(loader, model, optimizer, loss_fn, num_epochs):
    loop = tqdm(loader)

    # Send model to compute device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device=device)

    for epoch in range(num_epochs):
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

    # Saving model after last epoch
    save_checkpoint( model.state_dict(), 'checkpoints/model_checkpoint.pth' )