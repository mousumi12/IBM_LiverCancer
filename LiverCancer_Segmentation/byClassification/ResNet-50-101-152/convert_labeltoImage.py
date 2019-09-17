import os	
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt
import numpy as np
import gzip
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
import numpy as np
#import cPickle
import pickle
import csv
import re

dirname_labelImage = 'Prediction_label_Images/'
directory = os.path.dirname(dirname_labelImage)
if not os.path.exists(directory):
        os.makedirs(directory)


with open('dataset/resnet_v2_101_submission.csv', 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)

#print(your_list)
list_name = [i[0] for i in your_list]
list_label = [i[1] for i in your_list]
print(list_name)
print(list_label)

for idx,img in enumerate(list_name):
	parts = img.split('_')[:3]
	folder_name = '_'.join(parts) + '/'
	img_name = img.split('.jpg')[0] + '.tiff'
	image_folder = os.path.join(dirname_labelImage,folder_name)
	directory = os.path.dirname(image_folder)
	if not os.path.exists(directory):
		os.makedirs(directory)
	if re.match("tumor",list_label[idx]):
		new_image = Image.new('RGB', (512, 512), (255, 0, 0))
		#new_image = new_image.convert('1')
		new_image.save(os.path.join(dirname_labelImage,folder_name,img_name), "TIFF")

