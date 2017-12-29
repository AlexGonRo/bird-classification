import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.preprocessing.image import img_to_array
import os
from classifier.cnn_audio import CNN_audio
from utils.class_weights import class_weights
from keras.utils import plot_model
import time
import pickle
from keras.callbacks import ModelCheckpoint, TensorBoard

def imageGeneratorSugar(
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
        rescale=1. / 255,
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

def get_batches(
    dirname,
    gen=image.ImageDataGenerator(),
    shuffle=True,
    batch_size=16,
    class_mode='categorical',
    imageSizeTuple = (256,256),
    classes = None,
    color_mode = 'rgb',
    seed=2017):
    return gen.flow_from_directory(
        dirname,
        target_size=imageSizeTuple,
        class_mode=class_mode,
        shuffle=shuffle,
        classes = classes,
        batch_size=batch_size,
        color_mode = color_mode,
        seed=seed
    )

# Basically we  can shift sound horizontally and probably scale it a little bit, but rotations and vertical shifts are off-limits because in real life you cannot shift sound like this
genImage = imageGeneratorSugar(
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


# With 512 pictures per batch and about 100 epochs, it should achieve a decent accuracy.
batchSize = 128
train_path = "data/audio/train"
path_test = "data/audio/test"
save_model_path = "models/audio/"
save_pred_path = "results/audio/"
logs_path = "logs/audio/"
save_model = False
filters = 32
kernel_size = (3,3)
pool = (3,3)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
m,n = 64, 200
seed = 2017
epochs = 50
# lr = 1e-06
train_batches = get_batches(train_path, genImage, batch_size=batchSize, imageSizeTuple = (m,n),seed=seed)
valid_batches = get_batches(path_test, genImage, batch_size=batchSize, imageSizeTuple = (m,n), seed=seed)
class_weight = class_weights(train_path)
checkpoint = ModelCheckpoint(save_model_path + "my_model.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
logs_callback = TensorBoard(log_dir=logs_path, histogram_freq=0,
          write_graph=True, write_images=True)



model = CNN_audio(4, filters, kernel_size, pool, input_shape=(m,n,3))
model.compile()
#plot_model(model.model, to_file='model.png', show_layer_names=False, show_shapes=True)
# K.set_value(model.model.optimizer.lr, lr)
model.fit_generator(
    train_batches,
    test_generator=valid_batches,
    nb_epoch = epochs,
    class_weight = class_weight,
    callbacks = [early_stopping, checkpoint, logs_callback]
)

if save_model:
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    model.model.save_weights(filepath=save_model_path+'model2_128_epochs.h5')
    model.model.save(save_model_path+'model2_128_epochs.h5')  # creates a HDF5 file 'my_model.h5'

# Predict and save predictions for later use
#model.model.load_weights(save_model_path + "my_model.h5")
x_test = []
y_test = []
img_names = []
classes = os.listdir(path_test)
for fol in classes:
    imgfiles = os.listdir(path_test + '/' + fol)
    for img_name in imgfiles:
        im = Image.open(path_test + '/' + fol + '/' + img_name)
        im = im.convert(mode='RGB')
        im = im.resize((n,m))
        im = img_to_array(im) / 255
        x_test.append(im)
        y_test.append(fol)
        img_names.append(img_name)

x_test = np.array(x_test)
y_test = np.array(y_test)

predictions = model.predict(x_test)

# Save to pkl
my_dict = {'y_test':y_test, "predictions":predictions, "class_indices":valid_batches.class_indices}
pickle.dump(my_dict, open(save_pred_path + str(time.time()) + ".pkl", "wb" ))

if not os.path.exists(save_pred_path):
    os.makedirs(save_pred_path)

with open(save_pred_path + str(time.time()) + ".csv", "w") as f:
    label_map_im_train = (valid_batches.class_indices)
    for k, v in label_map_im_train.items():
        f.write(str(k) + ' >>> ' + str(v) + '\n')

    f.write("name,ground_truth,pred\n")
    for name, label, pred in zip(img_names, y_test, predictions):
        f.write(name + ","+ label + "," + str(pred) + "\n")
