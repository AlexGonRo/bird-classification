from scipy import misc
import os, os.path
import numpy as np
import h5py
from PIL import Image

from sklearn.model_selection import train_test_split

def get_imgs(species, use_cached=False, path_data="data/img/"):
    print("Loading images...")
    if not use_cached:
        data, labels = _load_data(species, path_data)
    else:
        # Load data
        hdf5_file = h5py.File(path_data + "all_img.h5", mode='w')
        data = hdf5_file["images"]
        labels = hdf5_file['classes']
        hdf5_file.close()
    indices = np.arange(len(data))
    print('Images loaded')
    print('Splitting')
    X_train, X_test, y_train, y_test, idx1, idx2 = train_test_split(data, labels, indices, test_size = 0.2,
                                                        random_state = 42, shuffle=True)

    return data, labels, idx1, idx2

'''
Load data and split into partitions for cross validation
@param load: if true entire data is loaded into memory
'''
def _load_data(species, path_data="data/img/"):
    imgs = []
    classes = []
    valid_images = [".jpg", ".gif", ".png", ".tga"]
    for i, specie in enumerate(species):
        path = path_data + specie
#        counter = 0
        for f in os.listdir(path):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            imgs.append(misc.imread(os.path.join(path, f)))
            bird_class = [0  if i!=j else 1 for j in range(len(valid_images))]
            classes.append(bird_class)
#            counter *=1
#            if counter > 5:
#                break

    # Save the data
    print("Saving to h5...")
    imgs_shape = (len(imgs), 224, 224, 3)
    hdf5_file = h5py.File(path_data + "all_img.h5", mode='w')
    hdf5_file.create_dataset("images", imgs_shape, dtype='uint8', data=imgs)
    hdf5_file.create_dataset("classes", (len(classes), 4), dtype='int32', data=classes)
    hdf5_file.close()

    return imgs, classes


def load_img_file(filename, num_classes):
    file = open(filename, "r")
    image_names = file.readlines()
    images = []
    labels = []
    c = 0
    for name in image_names:
        c +=1
        if c > 50:
            break
        label = int(name[:3])
        if label <= num_classes:
            im = Image.open("images/" + name.rstrip('\n'))
            H, W = im.size
            pixels = list(im.getdata())
            if not type(pixels[0]) is int:
                # todo: right now we are discarding transparent images
                image = np.array([comp for pixel in pixels for comp in pixel]).reshape(-1, H, W, 3)
                images.append(image)
                # zero-index the label
                labels.append(label - 1)
        else:
            break
    return images, labels