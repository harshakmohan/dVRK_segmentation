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

def train_fn(train_loader, val_loader, model, optimizer, loss_fn, num_epochs, checkpoint_name: str):

    # Send model to compute device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device=device)
    bnorm = nn.BatchNorm2d(num_features=3)

    for epoch in range(num_epochs):
        print(f'Epoch #{epoch}')
        with tqdm(train_loader) as loop:
            for batch_idx, (data, targets) in enumerate(loop):
                data = bnorm(data).to(device=device)
                targets = targets.float().unsqueeze(1).to(device=device)

                # forward
                with torch.cuda.amp.autocast():
                    predictions = torch.sigmoid(model(data))
                    loss = loss_fn(predictions, targets)
                    print(f'epoch #{epoch}; loss = {loss.item()}')

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())
        # Get a val accuracy here
        check_accuracy(val_loader, model)

    # Saving model after last epoch
    save_checkpoint( model.state_dict(), f'checkpoints/{checkpoint_name}.pth' )