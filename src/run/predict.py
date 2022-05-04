'''
Run inference on a dataset using a trained model. Define inference loop here
'''

# TODO: @Harsha Flesh out inference function
import numpy as np
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from src.models import UNET
from .utils import (load_checkpoint, save_checkpoint, check_accuracy, check_dice)
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

def save_best_worst(loader, model, device="cuda:0" if torch.cuda.is_available() else "cpu", folder="predictions/"):
    model.eval()

    img_dice = check_dice(loader, model, device)
    img_dice_tensor = torch.Tensor(img_dice)

    # After computing dice for each image, find the best and worst one and save it
    img_dice_tensor.cpu().numpy()
    index_max = np.argmax(img_dice_tensor)
    index_min = np.argmin(img_dice_tensor)

    dice_max = int(img_dice_tensor[index_max]*100)
    dice_min = int(img_dice_tensor[index_min]*100)

    ## Save original image, gt, and predicted seg mask of max index & min index

    for idx, (x, y) in enumerate(loader):
        if int(idx)==int(index_max):
            print('Found max val!')
            x = x.to(device)
            y = y.to(device)
            pred_best = torch.sigmoid(model(x)).float()
            pred_best = (pred_best > 0.5).float()

            torchvision.utils.save_image(pred_best, f"{folder}best_seg_pred_ds{dice_max}.png")
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}best_seg_gt.png")
            torchvision.utils.save_image(x, f"{folder}best_seg_original_image_index{idx}.png")

    for idx, (x, y) in enumerate(loader):
        if int(idx)==int(index_min):
            print('Found min val!')
            x = x.to(device)
            y = y.to(device)
            pred_worst = torch.sigmoid(model(x)).float()
            pred_worst = (pred_worst > 0.5).float()

            torchvision.utils.save_image(pred_worst, f"{folder}worst_seg_pred_ds{dice_min}.png")
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}worst_seg_gt.png")
            torchvision.utils.save_image(x, f"{folder}worst_seg_original_image_index{idx}.png")

    print('Done...')


