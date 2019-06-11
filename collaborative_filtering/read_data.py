import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
import sys
from ReadSave import *
import os


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

    data['user'].astype("category")
    data['artist'].astype("category")
    data['plays'].astype(float)

    
    return data

def get_indexes(triplets):
    print('Obtaining indexes...', end='')
    artist_index = {}
    index_artist = {}
    
    user_index = {}
    index_user = {}


    for (artistid, artistname) in  enumerate(triplets['artist'].cat.categories):
        artist_index[artistname] = artistid
        index_artist[artistid] =artistname
    
    for (userid,username) in enumerate(triplets['user'].cat.categories):
        user_index[username] = userid
        index_user[userid] = username
    
    return artist_index, index_artist, user_index, index_user
    
def get_plays(triplets):
    print('Obtaining plays...',end='')
    cols = triplets['user'].cat.codes.copy()
    rows = triplets['artist'].cat.codes.copy()
    data = triplets['plays'].astype(float)
    return coo_matrix((data, (rows, cols))).T


def split_data(dataset_path):

        print('Reading data...',end=' ')


        triplets_path = dataset_path + 'train_triplets_MSD-AG.txt'
        precomputed_path =  dataset_path + 'precomputed_data/' 


        full_data = read_triplets(triplets_path)
        train_data = get_train_data(full_data)

        artist_index, index_artist, user_index, index_user = get_indexes(full_data)
        plays_full  = get_plays(full_data)
        plays_train = get_plays(train_data)

        # save objects to cache them

        if not os.path.exists(precomputed_path):
            os.mkdir(precomputed_path)

        save_object( ( artist_index, index_artist,),  precomputed_path + 'artist_index_index_artist.pkl')
        save_object( (user_index, index_user),  precomputed_path + 'user_index_index_user.pkl')
        save_object( plays_full,  precomputed_path + 'plays_full.pkl')
        save_object( plays_train,  precomputed_path + 'plays_train.pkl')

        print('Done')
