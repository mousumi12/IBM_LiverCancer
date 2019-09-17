import os
import glob
import shutil
color_files = glob.glob('Datasets_fullMask/test/val/img/0/*.png')


label_files = ["Datasets_fullMask/test/val/gt/0/" + os.path.basename(m).split('color.png')[0] + 'labelFull.png' for m in color_files]

target_img = 'Datasets_fullMask/test2/val/img/0/'
target_gt = 'Datasets_fullMask/test2/val/gt/0/'


directory = os.path.dirname(target_img)
if not os.path.exists(directory):
	os.makedirs(directory)

directory = os.path.dirname(target_gt)
if not os.path.exists(directory):
	os.makedirs(directory)

idx = 0
for color_file, label_file in list(zip(color_files, label_files)): #[:40]:  #[:860]:
	#print(color_file, label_file)
	target_imgname = os.path.join(target_img,'{0:03d}_color.png'.format(idx))
	target_labelname = os.path.join(target_gt,'{0:03d}_labelBoundary.png'.format(idx))
	shutil.copyfile(color_file, target_imgname)
	shutil.copyfile(label_file, target_labelname)
	idx = idx + 1
