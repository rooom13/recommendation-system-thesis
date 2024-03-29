from ReadSave import *
from sklearn.preprocessing import normalize


def scale(plays):
       return normalize(plays) * 1000


def scale_data(datasetPath):
        print('Scaling data...', end=' ')
        precomputed_path = datasetPath + 'precomputed_data/'
        save_object( scale(read_object(precomputed_path + 'plays_train.pkl')), precomputed_path + 'norm_plays_train.pkl' )        
        save_object( scale(read_object(precomputed_path + 'plays_full.pkl')), precomputed_path + 'norm_plays_full.pkl' )        
        print('Done')
