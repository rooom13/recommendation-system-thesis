import pickle
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix, csr_matrix
from numpy import array
import numpy as np

# Just for a prettier matrix print
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

# load .pkl object
def read_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# load user_indices, artist_indices, plays 
def load_data():
    precomputed_path =  './fake_precomputed_data' if fakeDataset else './precomputed_data' 
    artist_user_path = precomputed_path + '/artist_user_indexes.pkl'
    plays_full_path = precomputed_path + '/plays_full.pkl'
    plays_train_path = precomputed_path + '/plays_train.pkl'

    artist_indices, user_indices = read_object(artist_user_path)
    plays_full = read_object(plays_full_path)
    plays_train = read_object(plays_train_path)

    return artist_indices, user_indices , plays_full, plays_train


fakeDataset = True

# load data from pickle files
artist_indices, user_indices , plays_full, plays_train = load_data()


plays_train = plays_train.tocsr().T
print(plays_train)
model = AlternatingLeastSquares
model.fit(plays_train)


for userid, username in enumerate(artist_indices):
    # write recommendation
    print(userid, username)




# user_vecs_reg, item_vecs_reg = implicit.alternating_least_squares(plays_train, factors=20, regularization = 0.1, iterations = 50)


