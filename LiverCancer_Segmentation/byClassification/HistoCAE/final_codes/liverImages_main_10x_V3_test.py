"""
Author : Mousumi Roy
"""

### Autoencoder with bottleneck block. HistoVAE architecture. V3: with added batchnormalization and relu activation for dense bottleneck.

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
# Data exploration
from keras.regularizers import l2

train_data = np.load('../datasets_Only10x/datasets_train/full_x.npy')
train_labels = np.load('../datasets_Only10x/datasets_train/full_y.npy')
val_data = np.load('../datasets_Only10x/datasets_val/full_x.npy')
val_labels = np.load('../datasets_Only10x/datasets_val/full_y.npy')
test_data = np.load('../datasets_Only10x/datasets_test/full_x.npy')
test_labels = np.load('../datasets_Only10x/datasets_test/full_y.npy')

# Shapes of training set
print("Training set (images) shape: {shape}".format(shape=train_data.shape))

print("Validation dataset (images) shape: {shape}".format(shape=val_data.shape))

# Shapes of test set
print("Test set (images) shape: {shape}".format(shape=test_data.shape))

train_data.dtype, test_data.dtype

# Create dictionary of target classes
label_dict = {
 0: 'non-viable',
 1: 'viable',
}

np.max(train_data), np.max(test_data)

train_X = train_data
valid_X = val_data
train_ground = train_labels
valid_ground = val_labels

# The convolutional Autoencoder
batch_size = 64 #128
epochs =  200 #50 # 100 #200
inChannel = 3
x, y = 256, 256
input_img = Input(shape = (x, y, inChannel))
num_classes = 2 
inner_dim = 2048  #1024  #2048 #4096
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


dec_ip = encoder(input_img)
enco = Model(input_img, dec_ip) 
enco.summary()

dec_ip = encoder(input_img)

autoencoder = Model(input_img, decoder(dec_ip))

adam = Adam(
            lr=lr, beta_1=beta_1
        )

autoencoder.compile(loss='mean_squared_error', optimizer = adam) #RMSprop())


autoencoder.summary()
autoencoder.layers
len(autoencoder.layers)


datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True
        )

cp_cb = ModelCheckpoint(filepath = '../checkpoints/autoencoder_bottleneck_withoutflatten_10x_V3.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')


"""
logger.debug('Fitting model')
recons_history = autoencoder.fit_generator(datagen.flow(train_X, train_X, batch_size=batch_size),
                    validation_data=(valid_X, valid_X), steps_per_epoch=len(train_X) / batch_size,
                    epochs=epochs,
                    callbacks=[cp_cb],
                    verbose=1,
                    shuffle=True)


#autoencoder_train = autoencoder.fit(train_X,train_X, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_X), shuffle = True)


loss = recons_history.history['loss']
val_loss = recons_history.history['val_loss']
epochs = range(200)
fig = plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
fig.savefig('Train_Val_Loss_reconst_bottleneck_withoutflatten_10x_V3.png')
#plt.show()
"""


# Show the reconstruction result from autoencoder
autoencoder.load_weights('../checkpoints/autoencoder_bottleneck_withoutflatten_10x_V3.h5')

score = autoencoder.evaluate(test_data, test_data, verbose=1)
print(score)
#f = open( 'file_recons_V2_withoutflatten_10x_V3.txt', 'a' )
#f.write( 'loss: = ' + repr(score) + '\n' )
#f.close()

liverImage_test = autoencoder.predict(test_data)
liverImage_val = autoencoder.predict(valid_X)

#print("Cifar10_test: {0}\nCifar10_val: {1}".format(np.average(liverImage_test), np.average(liverImage_val)))

def showOrigDec2(orig, dec, num=100):
    #from PIL import Image
    import scipy.misc
    n = num
    #resultDir = "Test_result_bottleneck_HistoVAE_V3_withoutflatten_10x_V3_200epoch/"
    #if not os.path.isdir(resultDir):
    #    os.makedirs(resultDir)
    for i in range(n):
        scipy.misc.imsave(resultDir + "Orig_file_{}.png".format(i), orig[i].reshape(256, 256, 3))
        scipy.misc.imsave(resultDir + "Recons_file_{}.png".format(i), dec[i].reshape(256, 256, 3))


#showOrigDec2(test_data, liverImage_test)


############## Segmenting the liver cancer images###############################

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_labels)
val_Y_one_hot = to_categorical(val_labels)
test_Y_one_hot = to_categorical(test_labels)

# Display the change for category label using one-hot encoding
#print('Original label:', train_labels[0])
#print('After conversion to one-hot:', train_Y_one_hot[0])

train_label = train_Y_one_hot
valid_label = val_Y_one_hot
train_X.shape,valid_X.shape,train_label.shape,valid_label.shape


def fc(enco):
    x = Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(enco)
    x = BatchNormalization()(x)

    flat = Flatten()(x)
    den = Dense(4096, activation='relu', kernel_regularizer=l2(0.01))(flat)
    den = BatchNormalization()(den)
    den = Dropout(0.7)(den)
    den = Dense(4096, activation='relu', kernel_regularizer=l2(0.01))(den)
    den = BatchNormalization()(den)
    den = Dropout(0.7)(den)
    den = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(den)
    den = BatchNormalization()(den)
    den = Dropout(0.7)(den)
    out = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(den)
    #out = Dense(1, activation='sigmoid')(den) 
    return out

encode = encoder(input_img)
full_model = Model(input_img,fc(encode))

for l1,l2 in zip(full_model.layers[:17],autoencoder.layers[0:17]):
    l1.set_weights(l2.get_weights())

full_model.get_weights()[0][1]
autoencoder.get_weights()[0][1]

for layer in full_model.layers[0:17]:
    layer.trainable = True #False

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            
        )
 
full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr =lr),metrics=['accuracy'])

# learning schedule callback
lrate = LearningRateScheduler(step_decay)

full_model.summary()
full_model.layers
len(full_model.layers)

############################# Train the model ######################################

def on_epoch_end(epoch, logs=None):
    print(K.eval(full_model.optimizer.lr))

cp_cb = ModelCheckpoint(filepath = '../checkpoints/autoencoder_classification_10x_V3.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

callbacks_list = [cp_cb, lrate]


#classify_train = full_model.fit(train_X, train_label, batch_size=64,epochs=100,verbose=1, callbacks=callbacks_list, validation_data=(valid_X, valid_label), shuffle = True)

#full_model.save_weights('autoencoder_classification_bottleneck_HistoVAE_V2_withoutflatten_10x_V3_200epoch.h5')

"""
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
fig.savefig('../Figures/Train_Val_accuracy_classification_10x_V3.png')

fig = plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
#plt.show()
fig.savefig('../Figures/Train_Val_Loss_classification_10x_V3.png')
"""
#############################################################
full_model.load_weights('../checkpoints/autoencoder_classification_10x_V3.h5')

# Model Evaluation on the Test Set
test_eval = full_model.evaluate(test_data, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
#f = open( 'file_classification_bottleneck_HistoVAE_V2_withoutflatten_10x_V3_200epoch.txt', 'a' )
#f.write( 'Test loss: = ' + repr(test_eval[0]) + '\n' )
#f.write( 'Test accuracy: = ' + repr(test_eval[1]) + '\n' )
#f.close()


# Predict labels
predicted_classes = full_model.predict(test_data)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
correct = np.where(predicted_classes==test_labels)[0]
print("Found %d correct labels" % len(correct))
#f = open( 'file_classification_bottleneck_HistoVAE_V2_withoutflatten_10x_V3_200epoch.txt', 'a' )
#f.write( 'correct labels: = ' + repr(len(correct)) + '\n' )
#f.close()


incorrect = np.where(predicted_classes!=test_labels)[0]
print("Found %d incorrect labels" % len(incorrect))
#f = open( 'file_classification_bottleneck_HistoVAE_V2_withoutflatten_10x_V3_200epoch.txt', 'a' )
#f.write( 'incorrect labels: = ' + repr(len(incorrect)) + '\n' )
#f.close()


# CLassification Report
from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_labels, predicted_classes, target_names=target_names))


