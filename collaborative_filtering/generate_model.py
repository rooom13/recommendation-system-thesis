from ReadSave import *
import implicit
from scipy.sparse import coo_matrix, csr_matrix


def generate_model(datasetPath,factors=64,regularization=0.1,iterations=50):
    precomputed_path = datasetPath + 'precomputed_data/'
    
    # load normalized data from pickle files
    plays_train = read_object(precomputed_path + '/norm_plays_train.pkl')
    
    model = implicit.als.AlternatingLeastSquares(factors=factors, iterations=iterations)
    model.fit(plays_train.T)
    save_object(model, precomputed_path + 'cf_model.pkl')

