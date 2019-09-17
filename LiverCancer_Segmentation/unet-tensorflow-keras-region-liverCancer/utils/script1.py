import numpy as np
import os, sys
from shutil import copyfile
import glob
from random import shuffle

label_files = glob.glob('../datasets_full/train/gt/0/*.png')
train_label_files = np.random.choice(label_files, 2000, replace=False)
train_label_files = train_label_files.tolist()
shuffle(train_label_files)

image_file_names = [os.path.basename(x).replace('_viable','') for x in train_label_files] 
train_color_files = ["../datasets_full/train/img/0/" + os.path.basename(m)  for m in image_file_names]


target_train_img = '../datasets/train/img/0/'
directory = os.path.dirname(target_train_img)
if not os.path.exists(directory):
	os.makedirs(directory)

target_train_gt = '../datasets/train/gt/0/'
directory = os.path.dirname(target_train_gt)
if not os.path.exists(directory):
        os.makedirs(directory)

target_val_img = '../datasets/val/img/0/'
directory = os.path.dirname(target_val_img)
if not os.path.exists(directory):
        os.makedirs(directory)

target_val_gt = '../datasets/val/gt/0/'
directory = os.path.dirname(target_val_gt)
if not os.path.exists(directory):
        os.makedirs(directory)

target_test_img = '../datasets/test/val/img/0/'
directory = os.path.dirname(target_test_img)
if not os.path.exists(directory):
        os.makedirs(directory)

target_test_gt = '../datasets/test/val/gt/0/'
directory = os.path.dirname(target_test_gt)
if not os.path.exists(directory):
        os.makedirs(directory)

for color_file, label_file in list(zip(train_color_files, train_label_files)):
	if os.path.isfile(color_file):
		target_imgname = os.path.join(target_train_img, os.path.basename(color_file))
		target_labelname = os.path.join(target_train_gt, os.path.basename(label_file))
		copyfile(color_file, target_imgname)
		copyfile(label_file, target_labelname)
	else:
		print(color_file)
		continue
label_files = []
image_file_names = []
label_files = glob.glob('../datasets_full/val/gt/0/*.png')
val_label_files = np.random.choice(label_files, 400, replace=False)
val_label_files = val_label_files.tolist()
shuffle(val_label_files)

image_file_names = [os.path.basename(x).replace('_viable','') for x in val_label_files]
val_color_files = ["../datasets_full/val/img/0/" + os.path.basename(m)  for m in image_file_names]

for color_file, label_file in list(zip(val_color_files, val_label_files)):
        target_imgname = os.path.join(target_val_img, os.path.basename(color_file))
        target_labelname = os.path.join(target_val_gt, os.path.basename(label_file))
        
        copyfile(color_file, target_imgname)
        copyfile(label_file, target_labelname)

label_files = []
image_file_names = []
test_label_files = glob.glob('../datasets_full/test/val/gt/0/*.png')

image_file_names = [os.path.basename(x).replace('_viable','') for x in test_label_files]
test_color_files = ["../datasets_full/test/val/img/0/" + os.path.basename(m)  for m in image_file_names]

for color_file, label_file in list(zip(test_color_files, test_label_files)):
        target_imgname = os.path.join(target_test_img, os.path.basename(color_file))
        target_labelname = os.path.join(target_test_gt, os.path.basename(label_file))

        copyfile(color_file, target_imgname)
        copyfile(label_file, target_labelname)

	
