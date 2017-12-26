# Given a set of images, it splits them into a training and test dataset and saves
# them accordingly.
import os
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from keras.preprocessing.image import  img_to_array

img_dir = "../data/img"
result_dir = "../data/img/"
m,n = 50,50
seed = 2017

classes=os.listdir(img_dir)
x=[]
y=[]
names = []
for fol in classes:
    imgfiles=os.listdir(img_dir+'/'+fol)
    for img in imgfiles:
        im=Image.open(img_dir+'/'+fol+'/'+img)
        x.append(im)
        y.append(fol)
        names.append(img)
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=seed)
    split_1, split_2 = sss.split(x, y)


train_index = split_1[0]
test_index = split_1[1]

x_train = [x[ind] for ind in train_index]
y_train = [y[ind] for ind in train_index]
names_train = [names[ind] for ind in train_index]
x_test = [x[ind] for ind in test_index]
y_test = [y[ind] for ind in test_index]
names_test = [[ind] for ind in test_index]

# Create training and test directory if they do not exist
if not os.path.exists(result_dir + "train"):
    os.makedirs(result_dir + "train")
if not os.path.exists(result_dir + "test"):
    os.makedirs(result_dir + "test")

# Save train index
for image, label, name in zip(x_train, y_train, names):
    try:
        image.save(result_dir + "train/" + label + "/" + name)
    except:
        os.makedirs(result_dir + "train/" + label)
        image.save(result_dir + "train/" + label + "/" + name)

# Save test index
for image, label, name in zip(x_test, y_test, names):
    try:
        image.save(result_dir + "test/" + label + "/" + name)
    except:
        os.makedirs(result_dir + "test/" + label)
        image.save(result_dir + "test/" + label + "/" + name)





