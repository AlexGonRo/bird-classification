import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def train_generator_img(data, labels, train_indx, batch_size):
    np.random.shuffle(train_indx)

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    num = 500
    while True:
        sampled_indx = np.random.choice(train_indx, size=num)
        d = []  # Sampled data
        l = []  # Sampled labels

        for i in sampled_indx:
            d.append(data[i])
            l.append(labels[i])
        d = np.array(d)
        l = np.array(l)

        datagen.fit(d)

        cnt = 0
        for X_batch, Y_batch in datagen.flow(d, l, batch_size=batch_size):
            weight = np.sum(Y_batch, axis=0) + 1
            weight = np.clip(np.log(np.sum(weight) / weight), 1, 5)
            weight = np.tile(weight, (len(Y_batch), 1))[Y_batch == 1]
            yield (X_batch, Y_batch, weight)
            cnt += batch_size
            if cnt == num:
                break


def val_generator_img(data, labels, val_indx, batch_size):
    np.random.shuffle(val_indx)

    start = 0
    while True:
        indx = val_indx[start:(start + batch_size) % len(val_indx)]
        start += batch_size
        if start > len(val_indx): start = 0

        yield (data[indx.sort()], labels[indx.sort()])