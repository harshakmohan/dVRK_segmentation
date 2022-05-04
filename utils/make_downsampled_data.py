from PIL import Image
from os import path as osp
from os import makedirs, mkdir, listdir
import cv2 as cv
import shutil

from cv2 import INTER_NEAREST

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

for i in [100,50,25,12,6]:
    dir_name = 'control' if i == 100 else f'scaled_{i}'
    mkdir(f'{resize_folder}/{dir_name}')
    mkdir(f'{resize_folder}/{dir_name}/img')
    mkdir(f'{resize_folder}/{dir_name}/labels')
    dir_pth = osp.join(resize_folder, dir_name)
    #resize each image by factor and save to benchmarking folder
    for img in listdir(raw_img):
        img_pth = osp.join(raw_img, img)
        mask_pth = osp.join(raw_masks, img)
        im = cv.imread(img_pth)
        mask = cv.imread(mask_pth)
        scale = i/100
        dim = (int(im.shape[1] * scale), int(im.shape[0] * scale))
        im_rsz = cv.resize(im, dim, interpolation=INTER_NEAREST)
        mask_rsz = cv.resize(mask, dim, interpolation=INTER_NEAREST)
        cv.imwrite(f'{dir_pth}/img/{img}', im_rsz)
        cv.imwrite(f'{dir_pth}/labels/{img}', mask_rsz)
        # im = Image.open(img_pth)
        # mask = Image.open(mask_pth)
        # width, height = im.size
        # im = im.resize((width//i, height//i))
        # mask = mask.resize((width//i, height//i))
        # im.save(f'{dir_pth}/img/{img}')
        # mask.save(f'{dir_pth}/labels/{img}')

