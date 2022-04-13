'''
Add any miscellaneous utility functions here
'''

import numpy as np
import torchvision
import logging
import torch

from torch.utils.data import DataLoader
from datasets import UCL, EndoVis
from torch import nn

__all__ =['save_checkpoint', 'load_checkpoint', 'check_accuracy', 'DiceLoss2D']

def save_checkpoint(state, filename='model_checkpoints/dummy_checkpoint.pth.tar'):
    print('=> Saving checkpoint')
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])


def check_accuracy(loader, model, device="cuda:0" if torch.cuda.is_available() else "cpu"):
    '''
    Check accuracy function for binary semantic segmentation.

    :param loader: Dataloader to check accuracy on. Requires access to ground truth segmentation mask.
    :param model: Model used to perform inference
    :param device: Compute device
    :return:
    '''

    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
    )
    print(f"Dice score: {dice_score / len(loader)}")
    model.train()


class DiceLoss2D(nn.Module):
    """Originally implemented by Cong Gao."""

    def __init__(self, skip_bg=False):
        super(DiceLoss2D, self).__init__()
        self.skip_bg = skip_bg

    def forward(self, inputs, target):
        # Add this to numerator and denominator to avoid divide by zero when nothing is segmented
        # and ground truth is also empty (denominator term).
        # Also allow a Dice of 1 (-1) for this case (both terms).
        eps = 1.0e-4

        if self.skip_bg:
            # numerator of Dice, for each class except class 0 (background)
            numerators = 2 * torch.sum(target[:, 1:] * inputs[:, 1:], dim=(2, 3)) + eps

            # denominator of Dice, for each class except class 0 (background)
            denominators = (
                    torch.sum(target[:, 1:] * target[:, 1:, :, :], dim=(2, 3))
                    + torch.sum(inputs[:, 1:] * inputs[:, 1:], dim=(2, 3))
                    + eps
            )

            # minus one to exclude the background class
            num_classes = inputs.shape[1] - 1
        else:
            # numerator of Dice, for each class
            numerators = 2 * torch.sum(target * inputs, dim=(2, 3)) + eps

            # denominator of Dice, for each class
            denominators = torch.sum(target * target, dim=(2, 3)) + torch.sum(inputs * inputs, dim=(2, 3)) + eps

            num_classes = inputs.shape[1]

        # Dice coefficients for each image in the batch, for each class
        dices = 1 - (numerators / denominators)

        # compute average Dice score for each image in the batch
        avg_dices = torch.sum(dices, dim=1) / num_classes

        # compute average over the batch
        return torch.mean(avg_dices)

