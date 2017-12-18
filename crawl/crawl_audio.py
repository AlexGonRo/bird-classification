from __future__ import print_function
import os.path
import os
from collections import defaultdict
import string
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.feature_extraction.text import CountVectorizer
import time
import pickle
from fake_useragent import UserAgent
from tqdm import tqdm


api_endpoint = 'http://www.xeno-canto.org/api/2/recordings'


def api_query(query=None, area=None, country=None, page=None):
    '''
    Creates an URI for quering the API.

    It does not ask the API for any data.

    Possible queries are:
    - country and page
    - area and page
    - Custom query and page
    '''
    if ((page is None) or (page == 0) or (page == '')):
        page = 1

    if ((query is None) or (query == '')):
        if ((area is None) or (area == '')):
            if ((country is None) or (country == '')):
                return None
            else:
                return api_endpoint + '?query=cnt:' + country + '&page=' + str(page)
        else:
            return api_endpoint + '?query=area:' + area + '&page=' + str(page)
    else:
        return api_endpoint + '?query=' + query + '&page=' + str(page)



# GET INFO ABOUT THE NUMBER OF RECORDINGs PER AREA

area_list = ['africa', 'america', 'asia', 'australia', 'europe']
area_df = pd.DataFrame(columns=['area', 'numRecordings', 'numSpecies', 'numPages'])
'''
for area in area_list:
    try:
        result = requests.get(api_query(area=area)) # All info about the recordings from an area. It also gives the info about the recordings of the first page of the area
        temp_dict = {'area': area,
                     'numRecordings': result.json()['numRecordings'],
                     'numSpecies': result.json()['numSpecies'],
                     'numPages': result.json()['numPages']}
        area_df = area_df.append(temp_dict, ignore_index=True)
    except Exception as ex:
        print('Failed to upload to ftp: ' + str(ex))

# GET SPECIFIC INFORMATION OF EACH RECORDING

ua = UserAgent()
headers = ua.firefox
headers = {'User-Agent': headers}
'''
recording_cols = ['cnt',    # Country
                 'date',    # Date
                 'en',      # Informal name of the bird
                 'file',    # URL to download
                 'gen', # Genus
                 'id',  # Sound ID
                 'lat', # Latitude
                 'lic', # License
                 'lng', # Longitude
                 'loc', # Specific location
                 'q',   # Score
                 'rec', # Author
                 'sp',  # Specie
                 'ssp',
                 'time',    # Time
                 'type',    # Type of call
                 'url',     # URL of the sound (but not to download)
                 'area',    # Continent
                 'page']
'''
result_df = pd.DataFrame(columns=[recording_cols])

# Create look up table to know which pages have already been visited
iter_df = pd.DataFrame(columns=['area','page','processed'])
for index, row in area_df.iterrows():
    iter_df_append = pd.DataFrame(columns=['area','page','processed'])
    iter_df_append.page = np.arange(1,row['numPages']+1,1)
    iter_df_append.processed = 0
    iter_df_append.area = row['area']
    iter_df = iter_df.append(iter_df_append,ignore_index =True)

# Ask for the recordings, one page at the time, and then mark the page as "visited" in the look up table
response_cols = ['cnt',
                   'date',
                   'en',
                   'file',
                   'gen',
                   'id',
                   'lat',
                   'lic',
                   'lng',
                   'loc',
                   'q',
                   'rec',
                   'sp',
                   'ssp',
                   'time',
                   'type',
                   'url']

result_df = pd.DataFrame(columns=[response_cols])

idx = np.arange(0, iter_df.shape[0])
for num in tqdm(idx):
    try:
        query_url = api_query(area=iter_df.iloc[num].area, page=int(iter_df.iloc[num].page))
        result = requests.get(query_url, headers=headers)

        temp_df = pd.DataFrame(result.json()['recordings'])
        temp_df['area'] = iter_df.iloc[num].area
        temp_df['page'] = int(iter_df.iloc[num].page)
        result_df = result_df.append(temp_df, ignore_index=True)

        iter_df.set_value(num, 'processed', 1)
        time.sleep(1)
        # Testing break
        # if (num==1):
        #    break
    except Exception as ex:
        print('Script failed for num: {}\nError type: {}\n'.format(str(num), str(ex)))


print(result_df.shape)

# Save scraped information to a file
result_df.to_csv('crawl_audio_info/bird_api_data.csv')
result_df[['id','file']].to_csv('crawl_audio_info/bird_files.csv')
'''
# We filter the records by the species we want
# TODO WE COULD ALSO FILTER BY TYPE OF BIRD SOUND

if "result_df" not in locals():
    result_df = pd.read_csv('crawl_audio_info/bird_api_data.csv')

#genuses = ["Turdus", "Phylloscopus", "Sylvia", "Emberiza"]
#species = ["migratorius", "sibilatrix", "communis", "citrinella" ]

genuses = ["Sylvia", "Poecile", "Phylloscopus", "Turdus"]     # Brown, Grey/White, Yellow/black, Red/black
species = ["communis", "atricapillus" ,"sibilatrix", "migratorius"]

# my_records_df = result_df.loc[result_df['gen'].isin(["Turdus", "Phylloscopus", "Sylvia", "Emberiza"])]
my_records_df = pd.DataFrame(columns=[recording_cols])

for gen, sp in zip (genuses, species):
    tmp = result_df[result_df.gen==gen]
    tmp = tmp[result_df.sp==sp]
    my_records_df = my_records_df.append(tmp)
    print("Genus: {}, specie: {}, instances: {}".format(gen, sp, tmp.shape))

print("Instances: {}".format(my_records_df.shape))

# If it crashed when crawling the data, comment the above code and write
#result_df = pd.read_csv('my_data.csv')

# print(my_genus_df.shape)

# print(my_genus_df.type.unique())

# print(my_genus_df["type"].value_counts())
# The most common types:

# We want to download the recordings


import os.path
import requests
import pandas as pd
import numpy as np
import time
import librosa
import signal

class TimeoutException(Exception):  # Custom exception class
    pass


def timeout_handler(signum, frame):  # Custom signal handler
    raise TimeoutException

def download_file(url, local_filename):
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                # f.flush() commented by recommendation from J.F.Sebastian
    return local_filename


c = 0   # Counter
cwd = os.getcwd()
partial_audio_data_f = "crawl_audio_info/part_audio_data.csv"
final_audio_data_f = "crawl_audio_info/audio_data.csv"
for index, row in my_records_df.iterrows():
#    if c < 248:
#        c += 1
#        continue
    file_path = row.file
    page = requests.get(file_path)
    file_path = page.url
    file_id = index

    dir_audio = 'bird_calls/'
    if not os.path.exists(dir_audio):
        os.makedirs(dir_audio)
    dir_features = 'sound_features/'
    if not os.path.exists(dir_features):
        os.makedirs(dir_features)
    audio_path = dir_audio + str(file_id) + '.mp3'
    mel_path = dir_features + 'mel_' + str(file_id)
    mfcc_path = dir_features + 'mfcc_' + str(file_id)   # Mel-frequency cepstrum
    # Change the behavior of SIGALRM
    signal.signal(signal.SIGALRM, timeout_handler)

    signal.alarm(15)
    try:
        # ! curl $file_path --output $audio_path
        download_file(file_path, audio_path)

        y, sr = librosa.load(audio_path)
        #sr as target sampling rate: the sample rate is the number of samples of a sound that are taken per second to represent the event digitally.
        my_records_df.set_value(index=file_id, col='length_s', value=y.shape[0] / sr)
        my_records_df.set_value(index=file_id, col='length', value=y.shape[0])
        my_records_df.set_value(index=file_id, col='sr', value=sr)
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=64) # Compute a mel-scaled spectrogram.
        log_S = librosa.logamplitude(S, ref_power=np.max)
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
        np.save(mel_path, log_S)
        np.save(mfcc_path, mfcc)
        my_records_df.set_value(index=file_id, col='mel_size', value=os.path.getsize(cwd + '/' + mel_path + '.npy'))
        my_records_df.set_value(index=file_id, col='mfcc_size', value=os.path.getsize(cwd + '/' + mfcc_path + '.npy'))
        my_records_df.set_value(index=file_id, col='processed', value=1)

    except TimeoutException:
        continue  # continue the for loop if function A takes more than 5 second
    else:
        # Reset the alarm
        signal.alarm(0)
        c = c + 1
        print(c)
        if c % 100 == 0: # Save from time to time in case the script fails.
            my_records_df.to_csv(partial_audio_data_f)

my_records_df.to_csv(final_audio_data_f)
os.remove(partial_audio_data_f)

