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

import random

def viz_test_data(dataloader, model):
    
    '''
    Function for visualizing predictions for testset

    Returns: None

    Parameter dataloader: dataloader object

    Parameter model: model for generating predictions
    '''     
    
    for data in train_dataloader:
        img, target = data
        
    img = img.detach().numpy()

    # This will change depending on what images we want to visualize
    y = img[random.randrange(10)]
    y = y.swapaxes(0,2)
    y = y.swapaxes(1,0)
    
    f, axarr = plt.subplots(1,2, figsize=(15,15))
    plt.axis('off')
    axarr[0].imshow(y)
    
    # Replace this with model prediction
    axarr[1].imshow(y)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);    

# viz_test_data(train_dataloader, model)
