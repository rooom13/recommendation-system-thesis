import pickle
import implicit
from scipy.sparse import coo_matrix, csr_matrix
from numpy import array
import numpy as np
import sys
import time
import metrics


# Just for a prettier matrix print
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})


def print_progress(text, completed, new_completed):
     if (new_completed > completed): 
            completed = new_completed
            sys.stdout.write('\r'+text+ str(completed) + ' %' )
            sys.stdout.flush()

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
     
    artist_user_path = precomputed_path + '/artist_user_indexes.pkl'
    plays_full_path = precomputed_path + '/norm_plays_full.pkl'
    plays_train_path = precomputed_path + '/norm_plays_train.pkl'

    artist_indices, user_indices = read_object(artist_user_path)
    plays_full = read_object(plays_full_path)
    plays_train = read_object(plays_train_path)

    return artist_indices, user_indices , plays_full, plays_train


def get_NDCG_list(k=5):
        ndcg_list = []
        completed = 0
        new_completed = 0
        for user_id in range(0,NUSERS):
                new_completed = (user_id +1)/ (NUSERS) * 100
                
                print_progress('Evaluating NDCG k=' + str(k) + '...  ', completed ,new_completed  )
                sr_rank = model.recommend(user_id, plays_train ) 
                scores = []
                for artist_id, x in sr_rank:
                        ground_truth = plays_full[user_id,artist_id]
                        if ground_truth != 0:
                                scores.append(ground_truth)
                ndcg_list.append(metrics.ndcg_at_k(scores, k))
        print(' ... completed', end='\n')
        return ndcg_list

# __MAIN__
fakeDataset = False
precomputed_path =  './fake_precomputed_data/' if fakeDataset else './precomputed_data/'

# load normalized data from pickle files
artist_indices, user_indices , plays_full, plays_train = load_data()
NUSERS,NARTISTS = plays_full.shape


# rows = items, cols = users
user_plays = plays_train.T.tocsr()

lazy = False and not fakeDataset
# Instantiate model or reuse cached
if not lazy:
        model = implicit.als.AlternatingLeastSquares(factors=20)
        model.fit(plays_train.T)
        save_object(model, precomputed_path + 'model.pkl')
else:
        model = read_object(precomputed_path +'model.pkl')




# print(plays_full.toarray().T)
# print(plays_train.toarray().T)

# METRICS NDCG


kk = [5,10,15]
for k in kk:
        NDCG_list = get_NDCG_list(k=k)
        print('NDCG_list_' + str(k),NDCG_list)
        save_object(NDCG_list,precomputed_path+'ndcg_'+str(k)+'.pkl')





# user_vecs_reg, item_vecs_reg = implicit.alternating_least_squares(plays_train, factors=20, regularization = 0.1, iterations = 50)
# artists= [artistname for  artist_id, artistname in enumerate(artist_indices)]


