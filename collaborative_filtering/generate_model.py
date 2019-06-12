from ReadSave import *
import implicit
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np

# load , artist_indices, plays 
def load_data(precomputed_path):
     
    plays_full_path = precomputed_path + '/norm_plays_full.pkl'
    plays_train_path = precomputed_path + '/norm_plays_train.pkl'

    plays_full = read_object(plays_full_path)
    plays_train = read_object(plays_train_path)

    return plays_full, plays_train

def generate_model(datasetPath,factors=64,regularization=0.1,iterations=50):
    precomputed_path = datasetPath + 'precomputed_data/'
    
    # load normalized data from pickle files
    plays_full, plays_train = load_data(precomputed_path)
    
    model = implicit.als.AlternatingLeastSquares(factors=factors, iterations=iterations)
    model.fit(plays_train.T)
    save_object(model, precomputed_path + 'cf_model.pkl')
    # user_vecs_reg, item_vecs_reg = implicit.alternating_least_squares(plays_train, factors=factors, regularization = regularization, iterations = iterations)

