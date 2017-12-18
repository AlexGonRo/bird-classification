#open cv
#import cv2
# installing open-cv is pain in the ass
# recently I found ubuntu package here
# otherwise use anaconda
# https://pypi.python.org/pypi/opencv-python
# sci-kit image
from skimage.transform import rescale
from skimage.transform import resize
import scipy.misc
import numpy as np
# A bulk of keras and theano imports
import theano
from theano import shared, tensor as T
from theano.tensor.nnet import conv2d, nnet
from theano.tensor.signal import pool

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping


batchSize = 16

# fit model
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

def fit(model, batches, val_batches, nb_epoch=1):
    model.fit_generator(
        batches,
        samples_per_epoch = batches.samples,
        nb_epoch=nb_epoch,
        validation_data=val_batches,
        nb_val_samples = val_batches.samples,
        callbacks = [early_stopping]
    )


def imageGeneratorSugar(
    featurewise_center,
    samplewise_center,
    featurewise_std_normalization,
    samplewise_std_normalization,
    rotation_range,
    width_shift_range,
    height_shift_range,
    shear_range,
    zoom_range,
    fill_mode='constant',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False):

    genImage = image.ImageDataGenerator(
        featurewise_center = featurewise_center,
        samplewise_center = samplewise_center,
        featurewise_std_normalization = featurewise_std_normalization,
        samplewise_std_normalization = samplewise_std_normalization,
        rotation_range = rotation_range,
        width_shift_range = width_shift_range,
        height_shift_range = height_shift_range,
        shear_range = shear_range,
        zoom_range =zoom_range,
        fill_mode = fill_mode,
        cval= cval,
        horizontal_flip = horizontal_flip,
        vertical_flip = vertical_flip)
    return genImage

def get_batches(
    dirname,
    gen=image.ImageDataGenerator(),
    shuffle=True,
    batch_size=batchSize,
    class_mode='categorical',
    imageSizeTuple = (256,256),
    classes = None,
    color_mode = 'rgb'
    ):
    return gen.flow_from_directory(
        dirname,
        target_size=imageSizeTuple,
        class_mode=class_mode,
        shuffle=shuffle,
        classes = classes,
        batch_size=batch_size,
        color_mode = color_mode
    )

# Basically we  can shift sound horizontally and probably scale it a little bit, but rotations and vertical shifts are off-limits because in real life you cannot shift sound like this
genImage = imageGeneratorSugar(
    featurewise_center = False,
    samplewise_center = False,
    featurewise_std_normalization = False,
    samplewise_std_normalization = False,
    rotation_range = 0,
    width_shift_range = 0.2,
    height_shift_range = 0,
    shear_range = 0,
    zoom_range = 0.1,
    fill_mode='constant',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False)


#  BASIC CNN MODEL
def getTestModelNormalize(inputShapeTuple, classNumber):
    model = Sequential([
            BatchNormalization(axis=1, input_shape = inputShapeTuple),
            Convolution2D(32, (3,3), activation='relu'),
            BatchNormalization(axis=1),
            MaxPooling2D((3,3)),
            Convolution2D(64, (3,3), activation='relu'),
            BatchNormalization(axis=1),
            MaxPooling2D((3,3)),
            Flatten(),
            Dense(200, activation='relu'),
            BatchNormalization(),
            Dense(classNumber, activation='softmax')
        ])

    model.compile(Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# With 512 pictures per batch and about 100 epochs, it should achieve a decent accuracy.
batchSize = 128*4
train_path = "data/audio/train"
valid_path = "data/audio/test"
train_batches = get_batches(train_path, genImage, batch_size=batchSize, imageSizeTuple = (64,200))
valid_batches = get_batches(valid_path, genImage, batch_size=batchSize,  imageSizeTuple = (64,200))
model2 = getTestModelNormalize(classNumber=132,inputShapeTuple=(64,200,3))
model2.optimizer.lr.set_value(1e-06)
model2.fit_generator(
    train_batches,
    steps_per_epoch = np.floor(train_batches.samples/batchSize),
    nb_epoch=30,
    validation_data=valid_batches,
    validation_steps = np.floor(valid_batches.samples/batchSize),
    callbacks = [early_stopping]
)

model2.save_weights(filepath=result_path+'model2_128_epochs.h5')
model2.save(result_path+'model2_128_epochs.h5')  # creates a HDF5 file 'my_model.h5'

