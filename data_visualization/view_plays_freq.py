
import matplotlib.pyplot as plt
import numpy as np
import sys

from ReadSave import read_object


def view_plays_freq(datasetPath, isNorm=True):

    precomputed_path = datasetPath +'precomputed_data/'
    userItem_path = precomputed_path + ('norm_plays_full.pkl' if isNorm else 'plays_full.pkl' )
    
    user_item = read_object(userItem_path).tocsr()

    data = np.round(user_item[user_item.nonzero()]).tolist()[0]
    plt.hist(data, bins = 100, edgecolor='black')
   

    if isNorm:
        plt.xlabel('Normalized plays count')
        plt.title('Histogram of normalized MSD-AG')
    else:
        plt.xlabel('Plays count')
        plt.title('Histogram of MSD-AG')
        plt.yscale('log')
    plt.ylabel('Rate')
    plt.grid(True)
    plt.show()

