from PIL import Image
from os import path as osp
from os import makedirs, mkdir, listdir
import shutil

#Using UCL dataset video 1
data_folder = osp.abspath('../data/ucl_dataset/Video_01')
raw_img = osp.join(data_folder, 'images')
raw_masks = osp.join(data_folder, 'ground_truth')

#check for directory and recreate if exists
resize_folder = osp.abspath('../data/benchmarking')
if not osp.exists(resize_folder):
    makedirs(resize_folder)
else:
    shutil.rmtree(resize_folder)
    makedirs(resize_folder)

for i in [1,2,4,8,16,32]:
    dir_name = 'control' if i == 1 else f'scaled_{i}'
    mkdir(f'{resize_folder}/{dir_name}')
    mkdir(f'{resize_folder}/{dir_name}/img')
    mkdir(f'{resize_folder}/{dir_name}/labels')
    dir_pth = osp.join(resize_folder, dir_name)
    #resize each image by factor and save to benchmarking folder
    for img in listdir(raw_img):
        img_pth = osp.join(raw_img, img)
        mask_pth = osp.join(raw_masks, img)
        im = Image.open(img_pth)
        mask = Image.open(mask_pth)
        width, height = im.size
        im = im.resize((width//i, height//i))
        mask = mask.resize((width//i, height//i))
        im.save(f'{dir_pth}/img/{img}')
        mask.save(f'{dir_pth}/labels/{img}')

