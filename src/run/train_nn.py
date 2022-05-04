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

def train_fn(train_loader, val_loader, model, optimizer, loss_fn, scheduler, num_epochs, checkpoint_name: str):

    # Send model to compute device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device=device)
    bnorm = nn.BatchNorm2d(num_features=3)
    dice = [0]

    for epoch in range(num_epochs):
        print(f'Starting epoch #{epoch+1}...')
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
        val_dice = check_accuracy(val_loader, model)

        # Save a model checkpoint if current state yields better dice score
        if val_dice > float(max(dice)):
            val_string = int(100*val_dice)
            save_checkpoint(state=model.state_dict(), filename=f'checkpoints/{checkpoint_name}_epoch{epoch}_{val_string}ds.pth')
        dice.append(val_dice)

        scheduler.step()
    # Saving model after last epoch
    val_string = int(100 * val_dice)
    save_checkpoint(state=model.state_dict(), filename=f'checkpoints/{checkpoint_name}_epoch{epoch}_{val_string}ds.pth' )

