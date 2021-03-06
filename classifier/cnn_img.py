from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


class CNN_img:

    def __init__(self, nb_classes, nb_filters, nb_conv, nb_pool, input_shape, last_layers=True):
        self.model = Sequential()
        self.model.add(Convolution2D(nb_filters, nb_conv, padding='same',
                                      input_shape=input_shape, activation="relu"))
        self.model.add(Convolution2D(nb_filters, nb_conv, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=nb_pool))
        self.model.add(Convolution2D(nb_filters, nb_conv, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=nb_pool))
#        self.model.add(Dropout(0.1))    # Applies Dropout to the input of the next layer
        self.model.add(Flatten())

#        self.model.add(Dropout(0.1))
        if last_layers:
            self.model.add(Dense(256))
            self.model.add(Dense(nb_classes, activation="softmax"))


    def compile(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',
                           metrics=['accuracy'])

    def fit(self):
        return self.model.fit()

    def fit_generator(self, train_generator, test_generator,
                    nb_epoch=10, class_weight=[], callbacks=[]):
        return self.model.fit_generator(train_generator, validation_data=test_generator,
                    epochs=nb_epoch, class_weight=class_weight, callbacks = callbacks)

    def predict(self, x_test):
        return self.model.predict(x_test)
