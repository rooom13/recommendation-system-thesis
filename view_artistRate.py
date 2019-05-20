
import matplotlib.pyplot as plt
import sys
from ReadSave import *


def print_histogram(x,bins=20, title='Histogram',xlabel='score', ylabel='# users' ):
        n, bins, patches = plt.hist(x, bins)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.yscale('log')
        plt.show()
        



fakeDataset =  False
full_plays_path = './precomputed_data/plays_full.pkl' if not fakeDataset else './fake_precomputed_data/plays_full.pkl'

plays = read_object(full_plays_path).tocsr()

NUSERS,NARTISTS = plays.shape

no0_artists = read_object('./precomputed_data/artistFreq.pkl')


for i in range(0,len(no0_artists)):
    if no0_artists[i] < 0:
        print(i,no0_artists[i])
# print(no0_artists)

# for userid in range(0,NUSERS):
#     x, nonzero = plays[userid].nonzero()
#     no0_artists.append(len(nonzero))
    
# print(no0_artists)

# save_object(no0_artists,'TATATA.pkl')

print_histogram(no0_artists,bins=50, title='Artist frequency',ylabel='# users',xlabel= '# artists')