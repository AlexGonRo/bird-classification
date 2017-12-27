from classifier.cnn_img import CNN_img
import os
from keras.preprocessing.image import ImageDataGenerator
import time
from keras.callbacks import EarlyStopping
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
from utils.class_weights import class_weights
from keras.utils import plot_model
import pickle

# input image dimensions
m,n = 112,112
path_train="data/img/train"
path_test="data/img/test"
classes=os.listdir(path_train)
batch_size=128
nb_classes=len(classes)
nb_epoch=20
nb_filters=32   # Number of filters
nb_pool= (3,3)
nb_conv= (3,3)   # Size of the convolution window
rs=2017
nb_epoch=50
save_model = False
save_model_path = "models/img/"
save_pred_path = "results/img/"
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
class_weight = class_weights(path_train)

gen_image = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
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

model = CNN_img(nb_classes, nb_filters, nb_conv, nb_pool, (n,m,3))
model.compile()
#plot_model(model.model, to_file='model.png', show_layer_names=False, show_shapes=True)
start = time.process_time()
model.fit_generator(im_generator_train, im_generator_test,
                    nb_epoch=nb_epoch, class_weight=class_weight, callbacks = [early_stopping])
end = time.process_time()

total_time = end-start

if save_model:
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    model.model.save_weights(filepath=save_model_path + 'model_img_{}b_{}epochs.h5'.format(batch_size, nb_epoch))
    model.model.save(save_model_path + 'model_img_{}b_{}epochs.h5'.format(batch_size, nb_epoch))

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
        im = img_to_array(im) / 255
        im = im.transpose(1,0,2)
        x_test.append(im)
        y_test.append(fol)
        img_names.append(img_name)

x_test = np.array(x_test)
y_test = np.array(y_test)

predictions = model.predict(x_test)

# Save to pkl
my_dict = {'y_test':y_test, "predictions":predictions, "class_indices":im_generator_train.class_indices}
pickle.dump(my_dict, open(save_pred_path + str(time.time()) + ".pkl", "wb" ))

if not os.path.exists(save_pred_path):
    os.makedirs(save_pred_path)

with open(save_pred_path + str(time.time()) + ".csv", "w") as f:
    label_map_im_train = (im_generator_train.class_indices)
    for k, v in label_map_im_train.items():
        f.write(str(k) + ' >>> ' + str(v) + '\n')

    f.write("name,ground_truth,pred\n")
    for name, label, pred in zip(img_names, y_test, predictions):
        f.write(name + ","+ label + "," + str(pred) + "\n")














        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

