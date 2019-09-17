# import the necessary packages
#from skimage.measure import structural_similarity as ssim
from skimage import measure
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
 
def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = measure.compare_ssim(imageA, imageB)
	print("mse:{}".format(m))
	print("ssim{}".format(s))
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
 
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the images
	plt.show()


# load the images -- the original, the original + contrast,
# and the original + photoshop
original = cv2.imread("Test_result_bottleneck_HistoVAE_V3_withoutflatten_10x_V3_300epoch_MSESSIMMAEloss/Orig_file_16.png")
recons_MSEloss = cv2.imread("Test_result_bottleneck_HistoVAE_V3_withoutflatten_10x_V3_200epoch/Recons_file_16.png")
recons_MSESSIMMAEloss = cv2.imread("Test_result_bottleneck_HistoVAE_V3_withoutflatten_10x_V3_300epoch_MSESSIMMAEloss/Recons_file_16.png")
recons_SSIMloss = cv2.imread("Test_result_bottleneck_HistoVAE_V3_withoutflatten_10x_V3_300epoch_SSIMloss/Recons_file_16.png")

# convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
recons_MSEloss = cv2.cvtColor(recons_MSEloss, cv2.COLOR_BGR2GRAY)
recons_MSESSIMMAEloss = cv2.cvtColor(recons_MSESSIMMAEloss, cv2.COLOR_BGR2GRAY)
recons_SSIMloss = cv2.cvtColor(recons_SSIMloss, cv2.COLOR_BGR2GRAY)

# initialize the figure
fig = plt.figure("Images")
images = ("Original", original), ("recons_MSEloss", recons_MSEloss), ("recons_MSESSIMMAEloss", recons_MSESSIMMAEloss), ("recons_SSIMloss", recons_SSIMloss)
 
# loop over the images
for (i, (name, image)) in enumerate(images):
	# show the image
	ax = fig.add_subplot(1, 4, i + 1)
	ax.set_title(name)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")
 
# show the figure
plt.show()
fig.savefig('Compare_reconstruct_Images.png')

 
# compare the images
compare_images(original, recons_MSEloss, "Original vs. recons_MSEloss")
compare_images(original, recons_MSESSIMMAEloss, "Original vs. recons_MSESSIMMAEloss")
compare_images(original, recons_SSIMloss, "Original vs. recons_SSIMloss")


