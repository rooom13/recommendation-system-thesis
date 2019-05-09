import pickle
from sklearn.preprocessing import normalize


# Storing and readind objects
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def read_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def scale(plays):
       return normalize(plays) * 1000

# save_object( norm_plays_train,  store_path + 'norm_plays_train.pkl')
print(1)
save_object( scale(read_object('./precomputed_data/plays_train.pkl')), './precomputed_data/norm_plays_train.pkl' )        
print(2)
save_object( scale(read_object('./precomputed_data/plays_full.pkl')), './precomputed_data/norm_plays_full.pkl' )        