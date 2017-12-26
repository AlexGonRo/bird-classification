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

    def __init__(self, classes, filters, kernel_size, pool, input_shape):
        self.model = Sequential([
            BatchNormalization(axis=1, input_shape= (input_shape[0], input_shape[1], 3)),
            Convolution2D(filters, kernel_size, activation='relu'),
            BatchNormalization(axis=1),
            MaxPooling2D(pool),
            Convolution2D(filters*2, kernel_size, activation='relu'),
            BatchNormalization(axis=1),
            MaxPooling2D(pool),
            Flatten(),
            Dense(200, activation='relu', name="Dense1"),
            BatchNormalization(),
            Dense(classes, activation='softmax', name="Dense2")
        ])

    def compile(self):
        self.model.compile(Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self):
        return self.model.fit()

    def fit_generator(self, train_generator, test_generator,
                    nb_epoch=10, class_weight=[], callbacks=[]):
        return self.model.fit_generator(train_generator, validation_data=test_generator,
                    epochs=nb_epoch, callbacks = callbacks)

    def predict(self, x_test):
        return self.model.predict(x_test)