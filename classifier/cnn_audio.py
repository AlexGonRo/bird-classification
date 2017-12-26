from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
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

class CNN_audio:

    def __init__(self, classes, filters, kernel_size, pool, input_shape, last_layers=True):
        if last_layers:
            self.model = Sequential([
                Convolution2D(filters, kernel_size, input_shape=input_shape, activation='relu'),
                Convolution2D(filters, kernel_size, input_shape=input_shape, activation='relu'),
                MaxPooling2D(pool),
                Convolution2D(filters, kernel_size, activation='relu'),

                MaxPooling2D(pool),
                Flatten(),
                Dense(256, activation='relu'),
                Dense(classes, activation='softmax')
            ])
        else:
            self.model = Sequential([
                Convolution2D(filters, kernel_size, input_shape=input_shape, activation='relu'),
                Convolution2D(filters, kernel_size, input_shape=input_shape, activation='relu'),
                MaxPooling2D(pool),
                Convolution2D(filters, kernel_size, activation='relu'),

                MaxPooling2D(pool),
                Flatten(),
            ])

    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self):
        return self.model.fit()

    def fit_generator(self, train_generator, test_generator,
                    nb_epoch=10, class_weight=[], callbacks=[]):
        return self.model.fit_generator(train_generator, validation_data=test_generator,
                    epochs=nb_epoch, class_weight=class_weight, callbacks = callbacks)

    def predict(self, x_test):
        return self.model.predict(x_test)