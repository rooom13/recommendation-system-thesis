import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
import sys
from ReadSave import *


float_formatter = lambda x: "%.4f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
def read_triplets(dataset_path):
    print('Reading triplets...',end='')
    triplets = pd.read_csv(dataset_path,sep='\t', names=['user','artist','plays'])

    triplets['user'] = triplets['user'].astype("category")
    triplets['artist'] = triplets['artist'].astype("category")
    triplets['plays'] = triplets['plays'].astype(float)
   
    return triplets

def get_train_data(triplets, P = 0.85):
    # p = proportion
    print('Obtaining training set...', end='')
    msk = np.random.rand(len(triplets)) < P

    data =  triplets[msk]
    # data['user'] = data['user'].astype("category")
    # data['artist'] = data['artist'].astype("category")
    # data['plays'] = data['plays'].astype(float)
    # print(data)

    data['user'].astype("category")
    data['artist'].astype("category")
    data['plays'].astype(float)

    
    return data

def get_indexes(triplets):
    print('Obtaining indexes...', end='')
    artists_index = {}
    users_index = {}


    for (artistid, artistname) in  enumerate(triplets['artist'].cat.categories):
        artists_index[artistname] = artistid
    
    for (userid,username) in enumerate(triplets['user'].cat.categories):
        users_index[username] = userid
    
    return artists_index, users_index
    
def get_plays(triplets):
    print('Obtaining plays...',end='')
    cols = triplets['user'].cat.codes.copy()
    rows = triplets['artist'].cat.codes.copy()
    data = triplets['plays'].astype(float)
    return coo_matrix((data, (rows, cols))).T


def read_data(fakeDataset = True):

        print('Reading data...',end=' ')

        output_filename = 'dataset_objects.pkl'

        dataset_path = '../fake_dataset/triplets.txt' if fakeDataset else '../dataset/train_triplets_MSD-AG.txt'
        store_path = './precomputed_data/' if not  fakeDataset else './fake_precomputed_data/'


        full_data = read_triplets(dataset_path)
        train_data = get_train_data(full_data)

        artists_index,users_index = get_indexes(full_data)
        plays_full  = get_plays(full_data)
        plays_train = get_plays(train_data)

        # save objects to cache them
        save_object( (artists_index,users_index),  store_path + 'artist_user_indexes.pkl')
        save_object( plays_full,  store_path + 'plays_full.pkl')
        save_object( plays_train,  store_path + 'plays_train.pkl')

        print('Done')
