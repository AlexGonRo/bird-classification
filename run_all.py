import numpy as np
# A bulk of keras and theano imports
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
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Flatten, Merge
import os

#  BASIC CNN MODEL
def my_model(inputShapeAudio, classNumber):
    audio_model = Sequential([
            BatchNormalization(axis=1, input_shape = inputShapeAudio),
            Convolution2D(32, (3,3), activation='relu'),
            BatchNormalization(axis=1),
            MaxPooling2D((3,3)),
            Convolution2D(64, (3,3), activation='relu'),
            BatchNormalization(axis=1),
            MaxPooling2D((3,3)),
            Flatten(),
            Dense(200, activation='relu', name="Dense1")
#            BatchNormalization(),
#            Dense(classNumber, activation='softmax', name="Dense2")
        ])
    img_model = Sequential([
        Convolution2D(32,3,3,border_mode='same',input_shape=(50,50,3)),
        Activation('relu'),
        Convolution2D(32,3,3),
        Activation('relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.5),
        Flatten(),
        Dense(128),
        Dropout(0.5)
#        Dense(classNumber),
#        Activation('softmax')
        ])

    final_model = Sequential()
    final_model.add(Merge([audio_model, img_model], mode='concat'))

    final_model.add(Dense(256))
    final_model.add(Dense(classNumber, activation='softmax'))

    #img_model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    #audio_model.compile(Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    final_model.compile(Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return final_model

def get_batches(
    dirnameAudio,
    dirnameImg,
    gen_audio=image.ImageDataGenerator(),
    gen_img=image.ImageDataGenerator(),
    shuffle=True,
    batch_size=16,
    class_mode='categorical',
    imageSizeAudio = (256,256),
    imageSizeImg = (50,50),
    classes = None,
    color_mode = 'rgb'
    ):
    gen_audio.flow_from_directory(
        dirnameAudio,
        target_size=imageSizeAudio,
        class_mode=class_mode,
        shuffle=shuffle,
        classes = classes,
        batch_size=batch_size,
        color_mode = color_mode
    )

    return

def audio_generator(
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

# Basically we  can shift sound horizontally and probably scale it a little bit, but rotations and vertical shifts are off-limits because in real life you cannot shift sound like this
gen_audio = audio_generator(
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

gen_image = ImageDataGenerator(
    rotation_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='constant',
    cval=0.)

audio_generator_train = gen_audio.flow_from_directory(
    'data/audio/train',
    class_mode=None,
    seed=2017)
im_generator_train = gen_image.flow_from_directory(
    'data/img/train',
    class_mode=None,
    seed=2017)

train_batches = zip(gen_audio, gen_image)

audio_generator_test = gen_audio.flow_from_directory(
    'data/audio/test',
    class_mode=None,
    seed=2017)
im_generator_test = gen_image.flow_from_directory(
    'data/img/test',
    class_mode=None,
    seed=2017)

valid_batches = zip(gen_audio, gen_image)


# With 512 pictures per batch and about 100 epochs, it should achieve a decent accuracy.
batchSize = 128*4

train_audio_path = "data/audio/train"
valid_audio_path = "data/audio/test"
train_img_path = "data/img/train"
valid_img_path = "data/img/test"
result_path = "models/all/"
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
#train_batches = get_batches(train_audio_path, train_img_path, gen_all,
#                            batch_size=batchSize, imageSizeAudio = (64,200), imageSizeImg = (50,50))
#valid_batches = get_batches(valid_audio_path, valid_img_path, gen_all,
#                            batch_size=batchSize,  imageSizeAudio = (64,200), imageSizeImg = (50,50))
model = my_model(classNumber=4, inputShapeAudio=(64,200,3), inputShapeImg=(50,50,3))
#model.optimizer.lr.set_value(1e-06)
model.fit_generator(
    train_batches,
    steps_per_epoch = np.floor(train_batches.samples/batchSize),
    nb_epoch=30,
    validation_data=valid_batches,
    validation_steps = np.floor(valid_batches.samples/batchSize),
    callbacks = [early_stopping]
)

if not os.path.exists(result_path):
    os.makedirs(result_path)
model.save_weights(filepath=result_path+'model2_128_epochs.h5')
model.save(result_path+'model2_128_epochs.h5')  # creates a HDF5 file 'my_model.h5'

