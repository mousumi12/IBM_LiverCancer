import os
import glob
import csv
import re
import numpy as np

img_label0 = glob.glob('../../final_dataset/Test/000/*.tiff')
img_label1 = glob.glob('../../final_dataset/Test/001/*.tiff')

image_name= []
test_labels = []
predicted_classes = []

csv_file = csv.reader(open('dataset/resnet_v2_101_submission.csv', "r"), delimiter=",")

for img in img_label0:
	image_name.append(os.path.basename(img))
	test_labels.append(0)
	csv_file = csv.reader(open('dataset/resnet_v2_101_submission.csv', "r"), delimiter=",")
	for row in csv_file:
    		#if current rows 2nd value is equal to input, print that row
    		if re.match(os.path.basename(img).split('.tiff')[0] +'.jpg', row[0]):
                        if re.match("tumor", row[1]):
                                predicted_classes.append(1)
                        else:
                                predicted_classes.append(0)


for img in img_label1:
	image_name.append(os.path.basename(img))
	test_labels.append(1)
	csv_file = csv.reader(open('dataset/resnet_v2_101_submission.csv', "r"), delimiter=",")
	for row in csv_file:
		if re.match(os.path.basename(img).split('.tiff')[0] +'.jpg', row[0]):
			if re.match("tumor",row[1]):
				predicted_classes.append(1)
			else:
				predicted_classes.append(0)
	
image_name = np.array(image_name)
test_labels = np.array(test_labels)
predicted_classes = np.array(predicted_classes)

print(image_name.shape)
print(test_labels.shape)
print(predicted_classes.shape)

np.save('image_name', image_name)
np.save('test_labels', test_labels)
np.save('predicted_classes', predicted_classes)	
