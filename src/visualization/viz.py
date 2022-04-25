# What I need so that I don't waste time running the model on data that was already passed to it:
#         - Training dataloader object
#         - Validation dataloader object
#         - Testing dataloader object
#         - Model predictions on training images with corresponding ground truth masks
#         - Model predictions on validation images with corresponding ground truth masks
#         - Class-wise Dice scores on training images (need to match order of training images)
#         - Class-wise Dice scores on validation images (need to match order of training images)

# What I plan to do with the data
#         - Modular function for visualizing images with Class-wise highest/lowest Dice scores (for training and validation data) that also outputs scores
#         - Function for visualizing random test images with predictions
#         - Function for "sliding" through images so that we can visualiza all images at once without having to display all of them at once (outputs scores when possible)

# If we want to compare the performance of different models together, I can cook up a function for that too (more if needed)

import numpy as np
import random
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

def viz_val_data(dataloader, model, disp_all):

    '''
    Function for visualizing predictions for validation/training set with known ground truth

    Returns: None

    Parameter dataloader: dataloader object

    Parameter model: model for generating predictions
    '''

    loop = tqdm(dataloader, disable = True)

    img = torch.cat([item[0] for item in loop],dim = 0)
    target = torch.cat([item[1] for item in loop],dim = 0)

    if disp_all == True:
        pass
    else:
        index = random.randrange(img.shape[0]-1)
        data = img[index].detach().numpy()
        data = np.expand_dims(data, axis = 0)
    with torch.no_grad():
        preds = torch.sigmoid(model(torch.tensor(data)))
        preds = (preds > 0.5).float()

    # Visualize image
    img = img.detach().numpy()
    y = img[index]
    y = y.swapaxes(0,2)
    y = y.swapaxes(1,0)

    f, axarr = plt.subplots(1,3, figsize=(15,15))
    plt.axis('off')
    axarr[0].imshow(y)
    axarr[0].set_title("Original Image", y = -0.1)

    # Visualize corresponding mask
    target = target.detach().numpy()
    y = target[index]
    axarr[1].imshow(y, cmap=plt.get_cmap('gray'))
    axarr[1].set_title("Ground Truth Segmentation", y = -0.1)

    # Visualize predicted mask
    y = preds.detach().numpy()
    y = y.swapaxes(0,2)
    y = y.swapaxes(1,0)
    y = np.squeeze(y).copy()
    axarr[2].imshow(y, cmap=plt.get_cmap('gray'))
    axarr[2].set_title("Predicted Segmentation", y = -0.1)

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plt.show()

def viz_test_data(dataloader, model, disp_all):

    '''
    Function for visualizing predictions for testset with known ground truth

    Returns: None

    Parameter dataloader: dataloader object

    Parameter model: model for generating predictions
    '''

    loop = tqdm(dataloader, disable = True)

    img = torch.cat([item[0] for item in loop],dim = 0)

    if disp_all == True:
        pass
    else:
        index = random.randrange(img.shape[0]-1)
        data = img[index].detach().numpy()
        data = np.expand_dims(data, axis = 0)
    with torch.no_grad():
        preds = torch.sigmoid(model(torch.tensor(data)))
        preds = (preds > 0.5).float()

    # Visualize image
    img = img.detach().numpy()
    y = img[index]
    y = y.swapaxes(0,2)
    y = y.swapaxes(1,0)

    f, axarr = plt.subplots(1,2, figsize=(15,15))
    plt.axis('off')
    axarr[0].imshow(y)
    axarr[0].set_title("Original Image", y = -0.1)

    # Visualize predicted mask
    y = preds.detach().numpy()
    y = y.swapaxes(0,2)
    y = y.swapaxes(1,0)
    y = np.squeeze(y).copy()
    axarr[1].imshow(y, cmap=plt.get_cmap('gray'))
    axarr[1].set_title("Predicted Segmentation", y = -0.1)

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plt.show()
