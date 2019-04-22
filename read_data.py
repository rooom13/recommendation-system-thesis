import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
import sys

def countLines(dataset_path):
    print('Counting lines')
    return sum(1 for line in open(dataset_path))

def read_triplets(dataset_path):
    counter = 0
    completed = 0

    users = np.array([])
    artists = np.array([])

    row =np.array([])
    col =np.array([])
    data =np.array([])


    for chunk in pd.read_csv(dataset_path, sep='\t',header=None, chunksize=CHUNKSIZE):
        #user and artist indices
        readUsers = chunk.iloc[:, 0].unique()
        users = np.unique(np.append(users, readUsers))
        readArtists = chunk.iloc[:, 1].unique()
        artists = np.unique(np.append(artists, readArtists))

        for i, tupla in chunk.iloc[:, :].iterrows():
            user, artist, play = tupla[0], tupla[1], tupla[2]
            iUser, iArtist = np.where(users == user)[0][0], np.where(artists == artist)[0][0]
            row = np.append(row,iUser)
            col = np.append(col,iArtist)
            data = np.append(data,play)
        
        # progress counter
        counter += CHUNKSIZE
        new_completed = int(round(float(counter)/Nlines * 100))
        print_progress('Reading dataset... ', completed, new_completed)
    print(' ... Done')
    

    # return users.tolist(), artists.tolist(), None
    # return 1,2,3
    return users.tolist(), artists.tolist(), csr_matrix( ( data, (col, row)), shape=(artists.size, users.size) )

def print_progress(text, completed, new_completed):
     if (new_completed > completed): 
            completed = new_completed
            sys.stdout.write('\r'+text+ str(completed) + ' %' )
            sys.stdout.flush()

# precoumputed 
p_Nlines = 25701407

fakeDataset = True 
lazy = not fakeDataset and True

dataset_path = './test_dataset/triplets.txt' if fakeDataset else './dataset/train_triplets_MSD-AG.txt'
CHUNKSIZE = 1000 if not fakeDataset else 2
Nlines = countLines(dataset_path)

users, artists, plays = read_triplets(dataset_path)

# sparse matrix
item_user_raw = np.array([
    # 0  1  2  users
    [1, 0, 1 ],  # artist0 
    [1, 0, 222 ],  # artist1 
    [1, 0, 143 ],  # artist2 
    [2, 0, 133 ],  # artist3 
    [1, 0, 0 ],  # artist4 
    [32, 0, 1132 ],  # artist5 
    [1, 1, 1 ],  # artist6 
    [0, 2, 0 ],  # artist7 
    [0, 143, 1 ],  # artist8 
    [0, 1, 1 ],  # artist9 
])


# convert to compressed sparse row matrix (csr_matrix)
item_user_data = csr_matrix(item_user_raw)

print(item_user_data)