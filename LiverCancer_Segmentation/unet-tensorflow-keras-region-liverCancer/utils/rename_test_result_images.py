import os
import glob
import shutil

#label_files = glob.glob('dataset_full/test/val/gt/0/*.png')

color_files = glob.glob('../Test_01_01_0113/Image_withIdx/*.tiff')
color_files2 = glob.glob('../checkpoints/model-66107/*.png')

for color_file2 in color_files2:
	idx2 = os.path.basename(color_file2).split('_')[0]
	for color_file in color_files:
		idx1 = os.path.basename(color_file).split('_')[0] 
		if int(idx1) == int(idx2):
			target_imgname = os.path.join('../Test_01_01_0113/pred/','{}'.format('01_01_0113_' + '_'.join(os.path.basename(color_file).split('_')[1:])))
			shutil.copyfile(color_file2, target_imgname)
