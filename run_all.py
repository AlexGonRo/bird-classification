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
from classifier.cnn_all import CNN_all
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.preprocessing.image import img_to_array
from utils.class_weights import class_weights
from PIL import Image
from keras.callbacks import TensorBoard



def two_input_generator(gen_1, gen_2):
    x1,y1 = gen_1.next()
    x2,y2 = gen_2.next()
    while True:
            yield [x1, x2], y1

batch_size = 128
train_audio_path = "data/audio/train/"
test_audio_path = "data/audio/test/"
train_img_path = "data/img/train/"
test_img_path = "data/img/test/"
save_model_path = "models/all/"
save_pred_path = "results/all/"
logs_path = "logs/all"
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
logs_callback = TensorBoard(log_dir=logs_path, histogram_freq=0,  
          write_graph=True, write_images=True)
class_weight = class_weights(train_audio_path)
save_model = False
epochs = 3 


# Image parameters
m_img, n_img = 112, 112
filters_img =32
pool_img = (3,3)
conv_img = (3,3)   # Size of the convolution window

# Audio parameters
m_audio, n_audio = 64, 200
filters_audio =32
pool_audio = (3,3)
conv_audio = (3,3)   # Size of the convolution window


gen_audio = image.ImageDataGenerator(
    rescale = 1./255,
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    rotation_range=0,
    width_shift_range=0.2,
    height_shift_range=0,
    shear_range=0.0,
    zoom_range=0.1,
    fill_mode='constant',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False)

gen_image = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    fill_mode='constant',
    cval=0.)

audio_generator_train = gen_audio.flow_from_directory(
    'data/audio/train',
    target_size=(m_audio,n_audio),
    class_mode="categorical",
    shuffle=True,
    batch_size=batch_size,
    color_mode = "rgb",
    seed=2017)
im_generator_train = gen_image.flow_from_directory(
    'data/img/train',
    target_size=(m_img,n_img),
    class_mode="categorical",
    shuffle=True,
    batch_size=batch_size,
    color_mode = "rgb",
    seed=2017)

train_batches = two_input_generator(audio_generator_train, im_generator_train)

audio_generator_test = gen_audio.flow_from_directory(
    'data/audio/test',
    target_size=(m_audio,n_audio),
    class_mode="categorical",
    shuffle=True,
    batch_size=batch_size,
    color_mode = "rgb",
    seed=2017)
im_generator_test = gen_image.flow_from_directory(
    'data/img/test',
    target_size=(m_img,n_img),
    class_mode="categorical",
    shuffle=True,
    batch_size=batch_size,
    color_mode = "rgb",
    seed=2017)

valid_batches = two_input_generator(audio_generator_test, im_generator_test)


model = CNN_all(4, filters_img, filters_audio, conv_img, conv_audio,
                 pool_img, pool_audio, (m_img, n_img, 3), (m_audio, n_audio, 3))
model.compile()
#plot_model(model.model, to_file='model.png', show_layer_names=False, show_shapes=True)
model.fit_generator(
    train_batches,
    test_generator=valid_batches,
    epochs=epochs,
    class_weight=class_weight,
    callbacks = [early_stopping, logs_callback]
)

if save_model:
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    model.model.save_weights(filepath=save_model_path+'model2_128_epochs.h5')
    model.model.save(save_model_path+'model2_128_epochs.h5')  # creates a HDF5 file 'my_model.h5'

# Predict and save predictions for later use
x_img_test = []
x_audio_test = []
y_test = []
img_names = []
classes = os.listdir(train_audio_path)
for fol in classes:
    imgfiles = os.listdir(test_img_path + '/' + fol)
    audiofiles = os.listdir(test_audio_path + '/' + fol)
    for img_name, audio_name in zip(imgfiles, audiofiles):
        # Load image
        im = Image.open(test_img_path + '/' + fol + '/' + img_name)
        im = im.convert(mode='RGB')
        im = im.resize((m_img, n_img))
        im = img_to_array(im) / 255
        im = im.transpose(1,0,2)
        # Load audio
        au = Image.open(test_audio_path + '/' + fol + '/' + audio_name)
        au = au.convert(mode='RGB')
        au = au.resize((m_audio, n_audio))
        au = img_to_array(au) / 255
        au = au.transpose(1,0,2)
        x_img_test.append(im)
        x_audio_test.append(au)
        y_test.append(fol)
        img_names.append(img_name)

x_img_test = np.array(x_img_test)
x_audio_test = np.array(x_audio_test)
y_test = np.array(y_test)

predictions = model.predict([x_audio_test, x_img_test])

if not os.path.exists(save_pred_path):
    os.makedirs(save_pred_path)

with open(save_pred_path + str(time.time()) + ".csv", "w") as f:
    label_map_im_train = (im_generator_train.class_indices)
    for k, v in label_map_im_train.items():
        f.write(str(k) + ' >>> '+ str(v) + '\n')
    
    f.write("name,ground_truth,pred\n")
    for name, label, pred in zip(img_names, y_test, predictions):
        f.write(name + ","+ label + "," + str(np.argmax(pred)) + "\n")
