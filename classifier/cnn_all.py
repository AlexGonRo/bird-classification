from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from classifier.cnn_img import CNN_img
from classifier.cnn_audio import CNN_audio
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.convolutional import *
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Merge
import os
from keras.utils import plot_model
import time


class CNN_all:

    def __init__(self, classes, filters_img, filters_audio, kernel_size_img, kernel_size_audio,
                 pool_img, pool_audio, input_shape_img, input_shape_audio):
        self.audio_model = CNN_audio(classes, filters_audio, kernel_size_audio, pool_audio, input_shape_audio, False)
        self.img_model = CNN_img(classes, filters_img, kernel_size_img, pool_img, input_shape_img, False)
        self.model = Sequential()
        self.model.add(Merge([self.audio_model.model, self.img_model.model], mode='concat'))
        self.model.add(Dense(512))
        self.model.add(Dense(64))
        self.model.add(Dense(classes, activation='softmax'))


    def compile(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',
                           metrics=['accuracy'])

    def fit(self):
        return self.model.fit()

    def fit_generator(self, train_generator, test_generator,
                    epochs=10, class_weight=[], callbacks=[]):
        return self.model.fit_generator(train_generator, validation_data=test_generator,
                    epochs=epochs, class_weight=class_weight, callbacks = callbacks, steps_per_epoch=29, validation_steps=5)

    def predict(self, x_test):
        return self.model.predict(x_test)
