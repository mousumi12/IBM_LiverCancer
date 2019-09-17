import os
import glob
import shutil

#label_files = glob.glob('dataset_full/test/val/gt/0/*.png')

color_files = glob.glob('../Test_01_01_0113/Image_withIdx/*.tiff')


target_img = '../datasets_01_01_0113/test/val/img/0/'
directory = os.path.dirname(target_img)
if not os.path.exists(directory):
        os.makedirs(directory)

for color_file in color_files:
	target_imgname = os.path.join('../datasets_01_01_0113/test/val/img/0/','{0:05d}.png'.format(int(os.path.basename(color_file).split('_')[0])))
	#target_labelname = os.path.join('datasets/test/val/gt/0/','{0:05d}.png'.format(idx))
	shutil.copyfile(color_file, target_imgname)
	#shutil.copyfile(label_file, target_labelname)
