from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D


class CNN_img:

    def __init__(self, nb_classes, nb_filters, nb_conv, nb_stride, nb_pool, input_shape):
        self.model = Sequential()
        self.model.add(Convolution2D(nb_filters, nb_conv, nb_stride, padding='same',
                                     input_shape=input_shape, activation="relu"))
        self.model.add(Convolution2D(nb_filters, nb_conv, nb_stride, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        self.model.add(Dropout(0.5))    # Applies Dropout to the input of the next layer
        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes, activation="softmax"))
        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                           metrics=['accuracy'])

    def fit(self):
        return self.model.fit()

    def fit_generator(self, train_generator, test_generator, batch_size=32,
                    nb_epoch=10, callbacks=[]):
        return self.model.fit_generator(train_generator, test_generator, batch_size=batch_size,
                    nb_epoch=nb_epoch, callbacks = callbacks)

    def predict(self, x_test):
        return self.model.predict(x_test)