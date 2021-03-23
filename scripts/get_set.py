# Download images and labels related to the validation/test set in the dataset

import os
import cv2
import shutil
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--inp', type = str, help = 'Input path.')
parser.add_argument('--out', type = str, help = 'Output path.')
parser.add_argument('--label', type = str, help = 'Image labels.')
opt = parser.parse_args()

print(opt)

file = open(opt.label)

valid_set = list()

for line in file.readlines():
    valid_set.append(line.split('/')[-1].split('\n')[0])

n_images = 0

try:
    os.mkdir(opt.out)
except:
    pass

for subdir, dirs, files in os.walk(opt.inp):
    for file in sorted(files):
        if file.endswith(('.jpg')) and file in valid_set:
            img = Image.open(opt.inp + file)
            img = img.convert('RGB')
            img.save(opt.out + os.sep + file)
            shutil.copy(opt.inp + os.sep + str(file).split('.')[0] + '.txt', opt.out)
            n_images += 1

print(n_images, 'images.')