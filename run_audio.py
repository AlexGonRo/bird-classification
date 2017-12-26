import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.convolutional import *
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
from keras import backend as K
import os
from classifier.cnn_audio import CNN_audio
from utils.class_weights import class_weights

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
    batch_size=16,
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


# With 512 pictures per batch and about 100 epochs, it should achieve a decent accuracy.
batchSize = 128*4
train_path = "data/audio/train"
valid_path = "data/audio/test"
save_model_path = "models/audio"
save_pred_path = "results/audio"
filters = 32
kernel_size = (3,3)
pool = (3,3)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
m,n = 64, 200
lr = 1e-06
train_batches = get_batches(train_path, genImage, batch_size=batchSize, imageSizeTuple = (m,n))
valid_batches = get_batches(valid_path, genImage, batch_size=batchSize,  imageSizeTuple = (m,n))
class_weight = class_weights(train_path)


model = CNN_audio(4, filters, kernel_size, pool, input_shape=(m,n,3))
K.set_value(model.moel.optimizer.lr, lr)
model.fit_generator(
    train_batches,
    train_generator=valid_batches,
    nb_epoch = 50,
    class_weight = class_weight,
    callbacks = [early_stopping]
)

if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
model.save_weights(filepath=save_model_path+'model2_128_epochs.h5')
model.save(save_model_path+'model2_128_epochs.h5')  # creates a HDF5 file 'my_model.h5'

