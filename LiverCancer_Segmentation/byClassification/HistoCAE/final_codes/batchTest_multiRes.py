import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #model will be trained on GPU 0

# Loading the data
import keras
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt
import numpy as np
import gzip
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.preprocessing import image as image_utils
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
import numpy as np
#import cPickle
import pickle

dirname_labelImage = 'Prediction_label_Images/'
directory = os.path.dirname(dirname_labelImage)
if not os.path.exists(directory):
        os.makedirs(directory)
"""
dirname = 'Prediction_probability_Images'
directory = os.path.dirname(dirname)
if not os.path.exists(directory):
        os.makedirs(directory)
"""
# Create dictionary of target classes
label_dict = {
 0: 'non-viable',
 1: 'viable',
}

batch_size = 64 #128
epochs =  50 # 100 #200
inChannel = 3
x, y = 256, 256
input_img = Input(shape = (x, y, inChannel))
num_classes = 2

def recons_encoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def recons_decoder(conv4):
    #decoder
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

def classifier_encoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4) # (small and thick)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool5) # (small and thick)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)

    return pool6

def fc(enco):
    flat = Flatten()(enco)
    den = Dense(4096, activation='relu')(flat)
    den = Dropout(0.5)(den)
    den = Dense(4096, activation='relu')(den)
    den = Dropout(0.5)(den)
    den = Dense(256, activation='relu')(den)
    den = Dropout(0.5)(den)
    out = Dense(num_classes, activation='softmax')(den)
    #out = Dense(1, activation='sigmoid')(den) 
    return out

autoencoder = Model(input_img, recons_decoder(recons_encoder(input_img)))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
encode = classifier_encoder(input_img)
full_model = Model(input_img,fc(encode))
for l1,l2 in zip(full_model.layers[:19],autoencoder.layers[0:19]):
	l1.set_weights(l2.get_weights())
full_model.get_weights()[0][1]
autoencoder.get_weights()[0][1]
for layer in full_model.layers[0:19]:
	layer.trainable = True #False
full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
full_model.load_weights('autoencoder_classification_fullTrain.h5')

def testPerImage(img, test_data, test_labels):
	parts = img.split('_')[:3]
	folder_name = '_'.join(parts) + '/'
	image_folder = os.path.join(dirname_labelImage,folder_name)
	directory = os.path.dirname(image_folder)
	if not os.path.exists(directory):
        	os.makedirs(directory) 

        # Segmenting the liver cancer images
	# Change the labels from categorical to one-hot encoding
	test_Y_one_hot = to_categorical(test_labels)
	
	# Predict labels
	predicted_classes_prob = full_model.predict(test_data)
	predicted_classes = np.argmax(np.round(predicted_classes_prob),axis=1)

	#Save a patch of size 256x256 with this value in disc	
	if predicted_classes[0] == 1:
		new_image = Image.new('RGB', (512, 512), (255, 0, 0))
		new_image.save(os.path.join(dirname_labelImage,folder_name,img), "TIFF")

inp_dir = '../../../final_dataset/Test/' #'liver_data'
target_size = (256, 256)

classes = os.listdir(inp_dir)

all_images = []
all_labels = []

i = 0
for idx, c in enumerate(classes):
    img_list = os.listdir(inp_dir + '/' + c)
    print(idx)
    print(c)
    img_list.sort()

    j = 0
    
    for img in img_list:
        # For each test image patch generate theimage array do the perdiction and convert back to image and save as image with prediction label for all pixels
        fname = inp_dir + '/' + c + '/' + img
        image = image_utils.load_img(fname).resize(target_size,Image.ANTIALIAS)
        image = np.array(image.getdata()).reshape(target_size[0], target_size[1], 3)
        image = image.astype('float32')/255
        #all_images.append(image)
        #all_labels.append(idx)
        test_data = np.array(image)
        test_data = np.expand_dims(test_data, axis=0)
        test_label = idx
        testPerImage(img,test_data, test_label)
    	
