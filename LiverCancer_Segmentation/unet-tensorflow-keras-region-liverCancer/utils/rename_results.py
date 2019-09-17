import os
import glob
import shutil

files = glob.glob('Results_Combined/*.png')

idx = 0
target_img = 'Total/img/'
directory = os.path.dirname(target_img)
if not os.path.exists(directory):
	        os.makedirs(directory)

for file_name in files:
	target_imgname = os.path.join(target_img,'{0:03d}_color.png'.format(int(os.path.basename(file_name).split('.png')[0])))
	shutil.copyfile(file_name, target_imgname)
	idx = idx+1


