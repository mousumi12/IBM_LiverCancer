import numpy as np
import os, sys
from shutil import copyfile
import glob
from random import shuffle

label_files = glob.glob('../../Segmentation-KenCode/dataset/256_ViableTumor_New/Train/*.tiff')
train_label_files = np.random.choice(label_files, 5000, replace=False)
train_label_files = train_label_files.tolist()
shuffle(train_label_files)

image_file_names = [os.path.basename(x).replace('_viable','') for x in train_label_files] 
train_color_files = ["../dataset_Unet/256/Train/" + os.path.basename(m)  for m in image_file_names]


target_train_img = '../datasets/train/img/0/'
target_train_gt = '../datasets/train/gt/0/'
target_val_img = '../datasets/val/img/0/'
target_val_gt = '../datasets/val/gt/0/'
target_test_img = '../datasets/test/val/img/0/'
target_test_gt = '../datasets/test/val/gt/0/'


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
label_files = glob.glob('../../Segmentation-KenCode/dataset/256_ViableTumor_New/Val/*.tiff')
val_label_files = np.random.choice(label_files, 600, replace=False)
val_label_files = val_label_files.tolist()
shuffle(val_label_files)

image_file_names = [os.path.basename(x).replace('_viable','') for x in val_label_files]
val_color_files = ["../dataset_Unet/256/Val/" + os.path.basename(m)  for m in image_file_names]

for color_file, label_file in list(zip(val_color_files, val_label_files)):
        target_imgname = os.path.join(target_val_img, os.path.basename(color_file))
        target_labelname = os.path.join(target_val_gt, os.path.basename(label_file))
        
        copyfile(color_file, target_imgname)
        copyfile(label_file, target_labelname)

label_files = []
image_file_names = []
test_label_files = glob.glob('../../Segmentation-KenCode/dataset/256_ViableTumor_New/Test/*.tiff')

image_file_names = [os.path.basename(x).replace('_viable','') for x in test_label_files]
test_color_files = ["../dataset_Unet/256/Test/" + os.path.basename(m)  for m in image_file_names]

for color_file, label_file in list(zip(test_color_files, test_label_files)):
        target_imgname = os.path.join(target_test_img, os.path.basename(color_file))
        target_labelname = os.path.join(target_test_gt, os.path.basename(label_file))

        copyfile(color_file, target_imgname)
        copyfile(label_file, target_labelname)

	
