from ReadSave import *
import implicit
from scipy.sparse import coo_matrix, csr_matrix
from numpy import array
import numpy as np

# load user_indices, artist_indices, plays 
def load_data(precomputed_path):
     
    artist_user_path = precomputed_path + '/artist_user_indexes.pkl'
    plays_full_path = precomputed_path + '/norm_plays_full.pkl'
    plays_train_path = precomputed_path + '/norm_plays_train.pkl'

    artist_indices, user_indices = read_object(artist_user_path)
    plays_full = read_object(plays_full_path)
    plays_train = read_object(plays_train_path)

    return artist_indices, user_indices , plays_full, plays_train

def generate_model(datasetPath,factors=64,regularization=0.1,iterations=50):
    precomputed_path = datasetPath + 'precomputed_data/'
    
    # load normalized data from pickle files
    artist_indices, user_indices , plays_full, plays_train = load_data(precomputed_path)
    
    model = implicit.als.AlternatingLeastSquares(factors=factors, iterations=iterations)
    model.fit(plays_train.T)
    save_object(model, precomputed_path + 'model.pkl')
    # user_vecs_reg, item_vecs_reg = implicit.alternating_least_squares(plays_train, factors=factors, regularization = regularization, iterations = iterations)

