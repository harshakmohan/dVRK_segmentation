from PIL import Image
import numpy as np
import os
import shutil


def crop_center(img, cropx, cropy):
    width, height = img.size 
    left = round((width - cropx)/2)
    top = round((height - cropy)/2)
    x_right = round(width - cropx) - left
    x_bottom = round(height - cropy) - top
    right = width - x_right
    bottom = height - x_bottom

    return img.crop((left, top, right, bottom))

binary_path = os.path.abspath('../data/binary_endovis')
dir_exists = os.path.exists(binary_path)

#check for directory and recreate if exists
if not dir_exists:
    os.makedirs(binary_path)
    os.mkdir(f'{binary_path}/img')
    os.mkdir(f'{binary_path}/labels')
else:
    shutil.rmtree(binary_path)
    os.makedirs(binary_path)
    os.mkdir(f'{binary_path}/img')
    os.mkdir(f'{binary_path}/labels')

endovis_path  = os.path.abspath('../data/endovis_dataset/')
subfolders = [f.path for f in os.scandir(endovis_path) if f.is_dir()]

for folder in subfolders: #instrument_dataset_x
    root = os.path.abspath(folder)
    frames = os.path.join(root, 'left_frames')
    gt = os.path.join(root, 'ground_truth')
    gt_folders = [f.path for f in os.scandir(gt) if f.is_dir()]
    for i in range(len(os.listdir(frames))): #for each image in dataset
        img_name = f'frame{str(i).zfill(3)}.png'
        save_name = f'{folder[-1]}_frame{str(i).zfill(3)}.png'
        masks = [os.path.join(gt, f'{tool}/{img_name}') for tool in gt_folders if tool != os.path.abspath(os.path.join(gt, 'Other_labels'))] #masks associated with frame
        images = [np.asarray(Image.open(mask)) for mask in masks]
        summed = sum(images)
        binary_mask = np.where(summed > 0, 255, 0)
        img_mask = Image.fromarray(binary_mask.astype('uint8'))
        img_mask = crop_center(img_mask, 701, 538)
        img_mask.save(f'{binary_path}/labels/{save_name}')

        orig = Image.open(f'{frames}/{img_name}')
        orig = crop_center(orig, 701, 538)
        orig.save(f'{binary_path}/img/{save_name}')
