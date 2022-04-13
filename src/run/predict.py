'''
Run inference on a dataset using a trained model. Define inference loop here
'''

# TODO: @Harsha Flesh out inference function
import torch
import torchvision
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
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()

