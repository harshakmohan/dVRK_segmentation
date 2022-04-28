'''
Add any miscellaneous utility functions here
'''

import numpy as np
import torchvision
import logging
import torch

from torch.utils.data import DataLoader
from src.datasets import UCL, BinaryEndoVis
from torch import nn
from torchvision import transforms

__all__ =['save_checkpoint', 'load_checkpoint', 'plot_loss', 'check_accuracy', 'DiceLoss2D']

def save_checkpoint(state, filename='model_checkpoints/dummy_checkpoint.pth.tar'):
    print('=> Saving checkpoint')
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint)


def plot_loss():
    pass

def init_weights(modelcurr):
    '''
    Initialize model weights
    :param modelcurr:
    :return:
    '''
    for m in modelcurr.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight.data)
            #torch.nn.init.xavier_uniform_(m.weight.data)
            #torch.nn.init.constant_(m.weight.data,0)
            #m.bias.data.zero_()

def check_accuracy(loader, model, device="cuda:0" if torch.cuda.is_available() else "cpu"):
    '''
    Check accuracy function for binary semantic segmentation.

    :param loader: Dataloader to check accuracy on. Requires access to ground truth segmentation mask.
    :param model: Model used to perform inference
    :param device: Compute device
    :return:
    '''
    print('Checking val accuracy...')
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            #x = x.to(device) # TODO: Normalize the image here
            #print('input size: ', x.size())
            r_mean = torch.mean(x[:, 0, :, :])
            g_mean = torch.mean(x[:, 1, :, :])
            b_mean = torch.mean(x[:, 2, :, :])
            mean = [r_mean, g_mean, b_mean]

            r_std = torch.std(x[:, 0, :, :])
            g_std = torch.std(x[:, 1, :, :])
            b_std = torch.std(x[:, 2, :, :])
            std = [r_std, g_std, b_std]
            transform_norm = transforms.Compose([transforms.Normalize(mean, std)])

            x = transform_norm(x).to(device)
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
    return dice_score/len(loader)

def dice(a, b):
    '''
    Originally implemented by Prabha Mandaleeka

    :param a: predicted segmentation mask
    :param b: ground truth segmentation mask
    :return: Sorensen-Dice coefficient between two masks
    '''
    return 100 * np.sum(a[b>0]) * 2 / (np.sum(a) + np.sum(b))


class SoftDiceLoss(nn.Module):

    def __int__(self):
        super(SoftDiceLoss, self).__int__()

    def forward(self, preds, ground_truth, epsilon=1e-6):
        # skip the batch and class axis for calculating Dice score
        axes = tuple(range(1, len(preds.shape) - 1))
        numerator = 2. * torch.sum(preds * ground_truth, dim=axes)
        denominator = torch.sum(torch.square(preds) + torch.square(ground_truth), dim=axes)
        return 1 - torch.mean(numerator / (denominator + epsilon))  # average over classes and batch


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

