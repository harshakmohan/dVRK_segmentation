from PIL import Image
import numpy as np
import os
import shutil


binary_path = os.path.abspath('../data/binary_endovis')
dir_exists = os.path.exists(binary_path)

#check for directory and recreate if exists
if not dir_exists:
    os.makedirs(binary_path)
else:
    shutil.rmtree(binary_path)
    os.makedirs(binary_path)

endovis_path  = os.path.abspath('../data/endovis_dataset/')
subfolders = [f.path for f in os.scandir(endovis_path) if f.is_dir()]

for folder in subfolders: #instrument_dataset_x
    root = os.path.abspath(folder)
    frames = os.path.join(root, 'left_frames')
    gt = os.path.join(root, 'ground_truth')
    gt_folders = [f.path for f in os.scandir(gt) if f.is_dir()]
    for i in range(len(os.listdir(frames))): #for each image in dataset
        img = f'frame{str(i).zfill(3)}.png'
        masks = [os.path.join(gt, f'{tool}/{img}') for tool in gt_folders if tool != os.path.abspath(os.path.join(gt, 'Other_labels'))] #masks associated with frame
        img_array = np.array()
        for mask in masks:
            img = np.asarray(Image.open(mask))
            img_array += img
        print(img_array) 
        break
    break
