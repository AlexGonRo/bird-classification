from __future__ import print_function
import os.path
from collections import defaultdict
import string
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.misc
import os
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer

import time

audio_info_path = '../crawl/crawl_audio_info/audio_data.csv'
sound_features_dir = '../crawl/sound_features/'
sound_features_cut_dir = '../crawl/sound_features_cut/'
audio_training_dir = '../data/audio/train/'
audio_test_dir = '../data/audio/test/'
data_dir = "../data/audio"

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
if not os.path.isdir(sound_features_cut_dir):
    os.mkdir(sound_features_cut_dir)
if not os.path.isdir(audio_training_dir):
    os.mkdir(audio_training_dir)
if not os.path.isdir(audio_test_dir):
    os.mkdir(audio_test_dir)

sample_df = pd.read_csv(audio_info_path)
sample_df[sample_df.processed==1].shape
# Plot length of the recordings
#sample_df[sample_df.processed==1].length_s.plot(kind='hist')

# Take only those audio files that are longer than 10s
cols = ['my_id', 'id', 'length', 'sr', 'mel_size', 'length_s', 'sp']
sub_sample_df = sample_df[(sample_df['length_s']>10)&(sample_df.processed==1)][cols]
sub_sample_df['w_id'] = ''
sub_sample_df.head()

# Test NVM for sound/noise decomposition

# Not implemented for now.

# Cut bird songs into sliding windows of 5s with 3s interval
result_list = []
for index, row in tqdm(sub_sample_df.iterrows()):
    my_id = int(row.my_id)
    try:
        song = np.load(sound_features_dir+'mel_' + str(my_id) + '.npy')
    except FileNotFoundError:
        print("Sound id {} not found".format(my_id))
        continue
    pointer = 0
    counter = 0
    song_length = song.shape[1]
    while (pointer < song_length):
        counter = counter + 1
        np.save(sound_features_cut_dir + str(my_id) + '_' + str(counter) + '.npy', song[:, pointer:pointer + 200])
        result_list.append([my_id, row.sp, row.length, row.sr, row.mel_size, row.length_s, counter])
        pointer = pointer + 300

#TODO Delete
with open("tmp.pkl", "wb") as f:
    pickle.dump(result_list, f)

#with open("tmp.pkl", "rb") as f:
#    result_list = pickle.load(f)

# Check how our dataset is now
# TODO Not implemented
print(len(result_list))

# Create training and validation set
# Also, transform the MEL into .jpeg
# TODO This last part could be changed and improved if necessary.


# Training and test set
X_train, X_test = train_test_split(
    result_list, test_size=0.2, random_state=2017)

# Train set
for element in tqdm(X_train):
    try:
        file_name = sound_features_cut_dir +str(element[0])+'_'+str(element[6])+'.npy'
        folder_name = str(element[1])
        new_file_name = audio_training_dir+folder_name+'/'+str(element[0])+'_'+str(element[6])+'.jpg'
        song = np.load(file_name)
        try:
            scipy.misc.imsave(new_file_name, song)
        except:
            os.mkdir(audio_training_dir+folder_name)
            scipy.misc.imsave(new_file_name, song)
    except:
        print(file_name)


# Validation set
for element in tqdm(X_test):
    try:
        file_name = sound_features_cut_dir +str(element[0])+'_'+str(element[6])+'.npy'
        folder_name = str(element[1])
        new_file_name = audio_test_dir+folder_name+'/'+str(element[0])+'_'+str(element[6])+'.jpg'
        song = np.load(file_name)
        try:
            scipy.misc.imsave(new_file_name, song)
        except:
            os.mkdir(audio_test_dir+folder_name)
            scipy.misc.imsave(new_file_name, song)
    except:
        print(file_name)
