import pickle
import implicit
from implicit.nearest_neighbours import bm25_weight
from scipy.sparse import coo_matrix, csr_matrix
from numpy import array
import numpy as np
import sys

# Just for a prettier matrix print
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})


# Storing and readind objects
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# load .pkl object
def read_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# load user_indices, artist_indices, plays 
def load_data():
    precomputed_path =  './fake_precomputed_data' if fakeDataset else './precomputed_data' 
    artist_user_path = precomputed_path + '/artist_user_indexes.pkl'
    plays_full_path = precomputed_path + '/norm_plays_full.pkl'
    plays_train_path = precomputed_path + '/norm_plays_train.pkl'

    artist_indices, user_indices = read_object(artist_user_path)
    plays_full = read_object(plays_full_path)
    plays_train = read_object(plays_train_path)

    return artist_indices, user_indices , plays_full, plays_train


fakeDataset = True

# load normalized data from pickle files
artist_indices, user_indices , plays_full, plays_train = load_data()




# rows = items, cols = users
plays_trainT = plays_train.T
user_plays = plays_train.T.tocsr()

lazy = True


# Instantiate model
if not lazy:
        model = implicit.als.AlternatingLeastSquares(factors=20)
        model.fit(plays_trainT)
        save_object(model, './fake_precomputed_data/model.pkl')
else:
        model = read_object('./fake_precomputed_data/model.pkl')



artists= [artistname for  artist_id, artistname in enumerate(artist_indices)]

# print(plays_full.toarray().T)
# print(plays_train.toarray().T)

# METRICS 
# Root Mean Squared Error
# RMSE


NUSERS,NARTISTS = plays_full.shape

for user_index in range(0,NUSERS):
        nonzero_artists =plays_full[user_index,:].nonzero()[1]
        for artist_index in  nonzero_artists:
                full_value, train_value =  plays_full[user_index,artist_index], plays_train[user_index,artist_index]
                if(train_value == 0):
                        print((user_index,artist_index), '-', (full_value,train_value) )
                        #  ? predicted_value = model.recommend(user_index, item_index) ?
                        # Compute RMSE 
                        # print(model.recommend(user_index, user_plays, N=2, ))



# user_vecs_reg, item_vecs_reg = implicit.alternating_least_squares(plays_train, factors=20, regularization = 0.1, iterations = 50)


