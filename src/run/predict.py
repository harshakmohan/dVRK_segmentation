'''
Run inference on a dataset using a trained model. Define inference loop here
'''

# TODO: @Harsha Flesh out inference function
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from src.models import UNET
from .utils import (load_checkpoint, save_checkpoint, check_accuracy)
from .utils import DiceLoss2D
import os



def save_predictions_as_imgs(loader, model, device, folder="predictions/"):
    # TODO: When saving prediction, save with Video_## and image ## in the name of the prediction image file.
    model.eval()
    for idx, (x, y) in enumerate(loader):
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

        with torch.no_grad():
            preds = torch.sigmoid(model(x)).float()
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
        if idx > len(loader)/3:
            break
    #model.train()

