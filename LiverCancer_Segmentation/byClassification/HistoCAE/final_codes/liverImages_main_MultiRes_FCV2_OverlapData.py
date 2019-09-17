import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5" #model will be trained on GPU 0

import keras
import math
#from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
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
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.layers import LeakyReLU, Lambda, Layer
from keras.preprocessing.image import ImageDataGenerator
import logging
logger = logging.getLogger(__name__)
import os
from keras.preprocessing import image as image_utils
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
import numpy as np
#import cPickle
import pickle
from keras.layers import *
from keras_contrib.losses import DSSIMObjective
from keras import metrics
from keras.regularizers import l2

# 20x dataset
train_data_20x = np.load('multires/datasets_Overlap/datasets_20x_train/full_x.npy')
train_labels_20x = np.load('multires/datasets_Overlap/datasets_20x_train/full_y.npy')
val_data_20x = np.load('multires/datasets_Overlap/datasets_20x_val/full_x.npy')
val_labels_20x = np.load('multires/datasets_Overlap/datasets_20x_val/full_y.npy')
test_data_20x = np.load('multires/datasets_Overlap/datasets_20x_test/full_x.npy')
test_labels_20x = np.load('multires/datasets_Overlap/datasets_20x_test/full_y.npy')


# 10x dataset
train_data_10x = np.load('multires/datasets_Overlap/datasets_10x_train/full_x.npy')
train_labels_10x = np.load('multires/datasets_Overlap/datasets_10x_train/full_y.npy')
val_data_10x = np.load('multires/datasets_Overlap/datasets_10x_val/full_x.npy')
val_labels_10x = np.load('multires/datasets_Overlap/datasets_10x_val/full_y.npy')
test_data_10x = np.load('multires/datasets_Overlap/datasets_10x_test/full_x.npy')
test_labels_10x = np.load('multires/datasets_Overlap/datasets_10x_test/full_y.npy')

# 5x dataset
train_data_5x = np.load('multires/datasets_Overlap/datasets_5x_train/full_x.npy')
train_labels_5x = np.load('multires/datasets_Overlap/datasets_5x_train/full_y.npy')
val_data_5x = np.load('multires/datasets_Overlap/datasets_5x_val/full_x.npy')
val_labels_5x = np.load('multires/datasets_Overlap/datasets_5x_val/full_y.npy')
test_data_5x = np.load('multires/datasets_Overlap/datasets_5x_test/full_x.npy')
test_labels_5x = np.load('multires/datasets_Overlap/datasets_5x_test/full_y.npy')

# Shapes of training set
print("Training set (images) shape: {shape}".format(shape=train_data_20x.shape))

print("Validation dataset (images) shape: {shape}".format(shape=val_data_20x.shape))

# Shapes of test set
print("Test set (images) shape: {shape}".format(shape=test_data_20x.shape))
print(test_labels_20x)
#print(test_data)
train_data_20x.dtype, test_data_20x.dtype

# Create dictionary of target classes
label_dict = {
 0: 'non-viable',
 1: 'viable',
}

train_X_20x = train_data_20x
valid_X_20x = val_data_20x
train_ground_20x = train_labels_20x
valid_ground_20x = val_labels_20x

train_X_10x = train_data_10x
valid_X_10x = val_data_10x
train_ground_10x = train_labels_10x
valid_ground_10x = val_labels_10x

train_X_5x = train_data_5x
valid_X_5x = val_data_5x
train_ground_5x = train_labels_5x
valid_ground_5x = val_labels_5x


# The convolutional Autoencoder

batch_size = 64
epochs = 200
inChannel = 3
x, y = 256, 256  #128, 128
input_img = Input(shape = (x, y, inChannel))
input_img1 = Input(shape = (x, y, inChannel))
input_img2 = Input(shape = (x, y, inChannel))
input_img3 = Input(shape = (x, y, inChannel))
num_classes = 2 
inner_dim = 2048 #1024 #512  #2048 #4096
dropout_rate = 0.5
lr= 0.0001
beta_1 = 0.05

def encoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, (3, 3), activation='relu', strides=(2, 2), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    #pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    #pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)

    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def decoder(conv4):
    #decoder
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    up1 = UpSampling2D((2,2))(conv5)

    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up2 = UpSampling2D((2,2))(conv6) #14 x 14 x 64

    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up3 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32

    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(up3) # 14 x 14 x 32
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    up4 = UpSampling2D((2,2))(conv8) # 28 x 28 x 32

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up4) # 28 x 28 x 1
    return decoded


dec_ip = encoder(input_img)  #, mask1, maks2, mask3, mask4, middle_dim, middle_tensor_shape 
enco = Model(input_img, dec_ip) 
enco.summary()
len(enco.layers)

########################
#20x autoencoder

autoencoder_20x = Model(input_img1, decoder(encoder(input_img1))) #, mask1, maks2, mask3, mask4, middle_dim, middle_tensor_shape))
#autoencoder_20x.summary()
autoencoder_20x.layers
len(autoencoder_20x.layers)


adam = Adam(
            lr=lr, beta_1=beta_1
        )



autoencoder_20x.compile(loss='mean_squared_error', optimizer = adam) #RMSprop())

######################
# 10x autoencoder

autoencoder_10x = Model(input_img2, decoder(encoder(input_img2))) #, mask1, maks2, mask3, mask4, middle_dim, middle_tensor_shape))
#autoencoder_10x.summary()
autoencoder_10x.layers
len(autoencoder_10x.layers)

adam = Adam(
            lr=lr, beta_1=beta_1
        )



autoencoder_10x.compile(loss='mean_squared_error', optimizer = adam) 
#####################
# 5x autoencoder

autoencoder_5x = Model(input_img3, decoder(encoder(input_img3))) #, mask1, maks2, mask3, mask4, middle_dim, middle_tensor_shape))
#autoencoder_5x.summary()
autoencoder_5x.layers
len(autoencoder_5x.layers)

adam = Adam(
            lr=lr, beta_1=beta_1
        )



autoencoder_5x.compile(loss='mean_squared_error', optimizer = adam) 

# reconstruction result from autoencoder

#  3 different reconstruction weights for 3 different resolutions
autoencoder_20x.load_weights('multires/autoencoder_bottleneck_withoutflatten_20x_V3_200epoch_OverlapData.h5')
autoencoder_10x.load_weights('multires/autoencoder_bottleneck_withoutflatten_10x_V3_200epoch_OverlapData.h5')
autoencoder_5x.load_weights('multires/autoencoder_bottleneck_withoutflatten_5x_V3_200epoch_OverlapData.h5')

score_20x = autoencoder_20x.evaluate(test_data_20x, test_data_20x, verbose=1)
print(score_20x)

score_10x = autoencoder_10x.evaluate(test_data_10x, test_data_10x, verbose=1)
print(score_10x)

score_5x = autoencoder_5x.evaluate(test_data_5x, test_data_5x, verbose=1)
print(score_5x)

#########################
# Segmenting the liver cancer images
# Change the labels from categorical to one-hot encoding
train_Y_one_hot_20x = to_categorical(train_labels_20x)
val_Y_one_hot_20x = to_categorical(val_labels_20x)
test_Y_one_hot_20x = to_categorical(test_labels_20x)
#test_Y_one_hot = test_Y_one_hot1[:100]

# Display the change for category label using one-hot encoding
print('Original label:', test_labels_20x[150])
print('After conversion to one-hot:', test_Y_one_hot_20x[150])

# Change the labels from categorical to one-hot encoding
train_Y_one_hot_10x = to_categorical(train_labels_10x)
val_Y_one_hot_10x = to_categorical(val_labels_10x)
test_Y_one_hot_10x = to_categorical(test_labels_10x)
#test_Y_one_hot = test_Y_one_hot1[:100]

# Display the change for category label using one-hot encoding
print('Original label:', test_labels_10x[150])
print('After conversion to one-hot:', test_Y_one_hot_10x[150])

# Change the labels from categorical to one-hot encoding
train_Y_one_hot_5x = to_categorical(train_labels_5x)
val_Y_one_hot_5x = to_categorical(val_labels_5x)
test_Y_one_hot_5x = to_categorical(test_labels_5x)
#test_Y_one_hot = test_Y_one_hot1[:100]

# Display the change for category label using one-hot encoding
print('Original label:', test_labels_5x[150])
print('After conversion to one-hot:', test_Y_one_hot_5x[150])

train_label_20x = train_Y_one_hot_20x
valid_label_20x = val_Y_one_hot_20x

#train_label = train_labels
#valid_label = val_labels
train_X_20x.shape,valid_X_20x.shape,train_label_20x.shape,valid_label_20x.shape

train_label_10x = train_Y_one_hot_10x
valid_label_10x = val_Y_one_hot_10x

#train_label = train_labels
#valid_label = val_labels
train_X_10x.shape,valid_X_10x.shape,train_label_10x.shape,valid_label_10x.shape

train_label_5x = train_Y_one_hot_5x
valid_label_5x = val_Y_one_hot_5x

#train_label = train_labels
#valid_label = val_labels
train_X_5x.shape,valid_X_5x.shape,train_label_5x.shape,valid_label_5x.shape

def fc(enco1, enco2, enco3):
    # enco1 for 20x
    
    x1 = Conv2D(128, (3, 3), activation='relu', strides=(2, 2), padding='same')(enco1)
    x1 = BatchNormalization()(x1)
    
    x1 = Conv2D(256, (3, 3), activation='relu', strides=(2, 2), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    
    x1 = Conv2D(512, (3, 3), activation='relu', strides=(2, 2), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    
    #x1 = Conv2D(512, (3, 3), activation='relu', strides=(2, 2), padding='same')(x1)
    #x1 = BatchNormalization()(x1)
   
    # enco1 for 10x
 
    x2 = Conv2D(128, (3, 3), activation='relu', strides=(2, 2), padding='same')(enco2)
    x2 = BatchNormalization()(x2)
    
    x2 = Conv2D(256, (3, 3), activation='relu', strides=(2, 2), padding='same')(x2)
    x2 = BatchNormalization()(x2)
    
    x2 = Conv2D(512, (3, 3), activation='relu', strides=(2, 2), padding='same')(x2)
    x2 = BatchNormalization()(x2)
    
    #x2 = Conv2D(512, (3, 3), activation='relu', strides=(2, 2), padding='same')(x2)
    #x2 = BatchNormalization()(x2)

    # enco1 for 5x
    
    x3 = Conv2D(128, (3, 3), activation='relu', strides=(2, 2), padding='same')(enco3)
    x3 = BatchNormalization()(x3)
    
    x3 = Conv2D(256, (3, 3), activation='relu', strides=(2, 2), padding='same')(x3)
    x3 = BatchNormalization()(x3)
    
    x3 = Conv2D(512, (3, 3), activation='relu', strides=(2, 2), padding='same')(x3)
    x3 = BatchNormalization()(x3)
    
    #x3 = Conv2D(512, (3, 3), activation='relu', strides=(2, 2), padding='same')(x3)
    #x3 = BatchNormalization()(x3)

    flat1 = Flatten()(x1)
    flat2 = Flatten()(x2)
    flat3 = Flatten()(x3)
    
    res = concatenate([flat1, flat2, flat3], axis = -1)
    
    den = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(res)
    den = BatchNormalization()(den)
    den = Dropout(0.5)(den)
    #den = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(den)
    #den = BatchNormalization()(den)
    #den = Dropout(0.3)(den)
    den = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(den)
    den = BatchNormalization()(den)
    den = Dropout(0.5)(den)
    out = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(den)
    #out = Dense(1, activation='sigmoid')(den) 
    return out

encode1 = encoder(input_img1)
encode2 = encoder(input_img2)
encode3 = encoder(input_img3)

full_model = Model([input_img1, input_img2, input_img3], fc(encode1, encode2, encode3))
full_model.summary()
len(full_model.layers)

# Load weights from reconstruction model

for l1,l2 in zip(full_model.layers[3:54:3],autoencoder_20x.layers[1:17]): #L[start:stop:step]
    l1.set_weights(l2.get_weights())
    
for l1,l2 in zip(full_model.layers[4:54:3],autoencoder_10x.layers[1:17]):
    l1
    l1.set_weights(l2.get_weights())
    
for l1,l2 in zip(full_model.layers[5:54:3],autoencoder_5x.layers[1:17]):
    l1
    l1.set_weights(l2.get_weights())


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
 
# learning schedule callback
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

#########################
# Classifying with full training (without frozen)

for layer in full_model.layers[0:54]:
    layer.trainable = True

def on_epoch_end(epoch, logs=None):
    print(K.eval(full_model.optimizer.lr))

# Train the Model

cp_cb = ModelCheckpoint(filepath = 'autoencoder_classification_bottleneck_HistoVAE_V2_withoutflatten_MultiRes_FCV2_300epoch_OverlapData.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
lr_decay = LearningRateScheduler(schedule=lambda epoch: lr * (0.9 ** epoch))
callbacks_list = [cp_cb, lr_decay]

full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer= Adam(lr=lr), metrics=['accuracy'])

on_epoch_end(epoch=10, logs=None)

"""
logger.debug('Fitting model')
classify_train = full_model.fit_generator(datagen.flow(train_X, train_label, batch_size=batch_size),
                    validation_data=(valid_X, valid_label), steps_per_epoch=len(train_X) / batch_size,
                    epochs= epochs,
                    callbacks=[cp_cb],
                    verbose=1,
                    shuffle=True)
"""

classify_train = full_model.fit([train_X_20x,train_X_10x,train_X_5x], train_label_20x, batch_size=64,epochs=200,verbose=1,callbacks=callbacks_list,validation_data=([valid_X_20x,valid_X_10x,valid_X_5x], valid_label_20x))

accuracy = classify_train.history['acc']
val_accuracy = classify_train.history['val_acc']
loss = classify_train.history['loss']
val_loss = classify_train.history['val_loss']
epochs = range(len(accuracy))
fig = plt.figure()
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
fig.savefig('Train_Val_accuracy_classification_bottleneck_HistoVAE_V2_withoutflatten_MultiRes_FCV2_300epoch_OverlapData.png')

fig = plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
#plt.show()
fig.savefig('Train_Val_Loss_classification_bottleneck_HistoVAE_V2_withoutflatten_MultiRes_FCV2_300epoch_OverlapData.png')

full_model.load_weights('autoencoder_classification_bottleneck_HistoVAE_V2_withoutflatten_MultiRes_FCV2_300epoch_OverlapData.h5')


# Model Evaluation on the Test Set
test_eval = full_model.evaluate([test_data_20x, test_data_10x,test_data_5x], test_Y_one_hot_20x, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
f = open( 'file_classification_bottleneck_HistoVAE_V2_withoutflatten_MultiRes_FCV2_300epoch_OverlapData.txt', 'a' )
f.write( 'Test loss: = ' + repr(test_eval[0]) + '\n' )
f.write( 'Test accuracy: = ' + repr(test_eval[1]) + '\n' )
f.close()


# Predict labels
predicted_classes = full_model.predict([test_data_20x, test_data_10x,test_data_5x])
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
correct = np.where(predicted_classes==test_labels_20x)[0]
print("Found %d correct labels" % len(correct))
f = open( 'file_classification_bottleneck_HistoVAE_V2_withoutflatten_multiRes_FCV2_300epoch_OverlapData.txt', 'a' )
f.write( 'correct labels: = ' + repr(len(correct)) + '\n' )
f.close()


incorrect = np.where(predicted_classes!=test_labels_20x)[0]
print("Found %d incorrect labels" % len(incorrect))
f = open( 'file_classification_bottleneck_HistoVAE_V2_withoutflatten_MultiRes_FCV2_300epoch_OverlapData.txt', 'a' )
f.write( 'incorrect labels: = ' + repr(len(incorrect)) + '\n' )
f.close()


# CLassification Report
from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_labels_20x, predicted_classes, target_names=target_names))

