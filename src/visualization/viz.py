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
