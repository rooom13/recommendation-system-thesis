import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
import sys
import pickle

float_formatter = lambda x: "%.0f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def read_triplets(dataset_path):

    triplets = pd.read_csv(dataset_path,sep='\t', names=['user','artist','plays'])

    triplets['user'] = triplets['user'].astype("category")
    triplets['artist'] = triplets['artist'].astype("category")
    triplets['plays'] = triplets['plays'].astype(float) # o int
   
    return triplets

def get_train_data(triplets, P = 0.85):
    # p = proportion
    msk = np.random.rand(len(triplets)) < P
    data =  triplets[msk]
    data['user'] = data['user'].astype("category")
    data['artist'] = data['artist'].astype("category")
    data['plays'] = data['plays'].astype(float)
    return data

def get_indexes(triplets):
    artists_index = {}
    users_index = {}


    for (artistid, artistname) in  enumerate(triplets['artist'].cat.categories):
        artists_index[artistname] = artistid
    
    for (userid,username) in enumerate(triplets['user'].cat.categories):
        users_index[username] = userid
    
    return artists_index, users_index
    
def get_plays(triplets):
    cols = triplets['user'].cat.codes.copy()
    rows = triplets['artist'].cat.codes.copy()
    data = triplets['plays'].astype(float)
    return coo_matrix((data, (rows, cols))).T

# Storing and readind objects
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def read_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


fakeDataset = True
output_filename = 'dataset_objects.pkl'

dataset_path = '../fake_dataset/triplets.txt' if fakeDataset else '../dataset/train_triplets_MSD-AG.txt'


full_data = read_triplets(dataset_path)
train_data = get_train_data(full_data)

artists_index,users_index = get_indexes(full_data)
plays_full  = get_plays(full_data)
plays_train = get_plays(train_data)

store_path = './precomputed_data/' if not  fakeDataset else './fake_precomputed_data/'

print(artists_index)
print(users_index)
save_object( (artists_index,users_index),  store_path + 'artist_user_indexes.pkl')



save_object( plays_full,  store_path + 'plays_full.pkl')
save_object( plays_train,  store_path + 'plays_train.pkl')





# item_user_raw = np.array([
#     # 0  1  2  users
#     [1, 0, 1 ],  # artist0 
#     [1, 0, 222 ],  # artist1 
#     [1, 0, 143 ],  # artist2  
#     [2, 0, 133 ],  # artist3 
#     [1, 0, 0 ],  # artist4 
#     [32, 0, 1132 ],  # artist5 
#     [1, 1, 1 ],  # artist6 
#     [0, 2, 0 ],  # artist7 
#     [0, 143, 1 ],  # artist8 
#     [0, 1, 1 ],  # artist9 
#     ])
# item_user_data = coo_matrix(item_user_raw)

