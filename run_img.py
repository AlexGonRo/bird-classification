from classifier.cnn_img import CNN_img
import os
from keras.preprocessing.image import ImageDataGenerator
import time
from keras.callbacks import EarlyStopping
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array

# input image dimensions
m,n = 50,50
path_train="data/img/train"
path_test="data/img/test"
classes=os.listdir(path_train)
batch_size=32
nb_classes=len(classes)
nb_epoch=20
nb_filters=32   # Number of filters
nb_pool= (2,2)
nb_conv= (3,3)   # Size of the convolution window
nb_stride=1 # How much the convolution window moves.
rs=2017
nb_epoch=5
batch_size=5
save_model = False
save_model_path = "models/img"
save_pred_path = "results/img"

gen_image = ImageDataGenerator(
    rotation_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='constant',
    cval=0.)

im_generator_train = gen_image.flow_from_directory(
    path_train,
    target_size=(m,n),
    class_mode="categorical",
    shuffle=True,
    batch_size=batch_size,
    color_mode = "rgb",
    seed=rs)

im_generator_test = gen_image.flow_from_directory(
    path_test,
    target_size=(m,n),
    class_mode="categorical",
    shuffle=True,
    batch_size=batch_size,
    color_mode = "rgb",
    seed=rs)

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model = CNN_img(nb_classes, nb_filters, nb_conv, nb_stride, nb_pool, (n,m,3))
start = time.process_time()
model.fit_generator(im_generator_train, im_generator_test, batch_size=batch_size,
                    nb_epoch=nb_epoch, callbacks = [early_stopping])
end = time.process_time()

total_time = start - end

if save_model:
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    model.save_weights(filepath=save_model_path + 'model_img_128_epochs.h5')
    model.save(save_model_path + 'model2_128_epochs.h5')  # creates a HDF5 file 'my_model.h5'

print("Total CPU time spent: {}".format(total_time))

# Predict and save predictions for later use
x_test = []
y_test = []
img_names = []
classes = os.listdir(path_test)
for fol in classes:
    imgfiles = os.listdir(path_test + '/' + fol)
    for img_name in imgfiles:
        im = Image.open(path_test + '/' + fol + '/' + img_name)
        im = im.convert(mode='RGB')
        im = im.resize((m, n))
        im = img_to_array(im)
        x_test.append(im)
        y_test.append(fol)
        img_names.append(img_name)

x_test = np.array(x_test)
y_test = np.array(y_test)

predictions = model.predict(x_test)

if not os.path.exists(save_pred_path):
    os.makedirs(save_pred_path)

with open(save_pred_path + str(time.time()), "w") as f:
    f.write("name,ground_truth,pred\n")
    for name, label, pred in zip(img_names, y_test, predictions):
        f.write(name + ","+ label + "," + str(pred))














        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

