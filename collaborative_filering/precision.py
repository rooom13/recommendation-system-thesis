from ReadSave import *
import implicit
from scipy.sparse import coo_matrix, csr_matrix
from numpy import array
import numpy as np
import sys
import metrics


# Just for a prettier matrix print
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})


def print_progress(text, completed, new_completed):
     if (new_completed > completed): 
            completed = new_completed
            sys.stdout.write('\r'+text+ str(completed) + ' %' )
            sys.stdout.flush()



# load user_indices, artist_indices, plays 
def load_data(precomputed_path):
     
    # use raw matrix (no scaled)
    plays_full_path = precomputed_path + 'plays_full.pkl'
    plays_train_path = precomputed_path + 'norm_plays_train.pkl'

    model_path = precomputed_path + 'model.pkl'
 
    plays_full = read_object(plays_full_path).tocsr()
    plays_train = read_object(plays_train_path)
    model = read_object(model_path)

    return plays_full, plays_train, model

def get_precision_list( plays_full, plays_train,model, k=5):


        NUSERS,NARTISTS = plays_full.shape


        precision_list = []
        completed = 0
        new_completed = 0

        # for each user
        for user_id in range(0,NUSERS):
                new_completed = (user_id +1)/ (NUSERS) * 100
                
                print_progress('Evaluating precision k=' + str(k) + '...  ', completed ,new_completed  )
                
                # obtain recommended k top
                sr_rank = model.recommend(user_id, plays_train, N=k ) 
                rellevants = []
                # for each recommended artist
                for artist_id, x in sr_rank:
                        ground_truth = plays_full[user_id,artist_id]
                        print(artist_id,ground_truth)
                        # if its ground truth is rellevant ( # plays > 0 ) add relevant (1) else non-relevant (0)
                        if ground_truth > 1:
                                rellevants.append(1)
                        else:
                                rellevants.append(0)
                # Compute precision of rellevants
                precision_list.append(metrics.precision_at_k(rellevants, k))
        print(' ... completed', end='\n')
        return precision_list

def evaluate_precision(fakeDataset):
        precomputed_path =  './fake_precomputed_data/' if fakeDataset else './precomputed_data/'

        # load normalized data from pickle files
        plays_full, plays_train,model = load_data(precomputed_path)

        kk = [5]
        # kk = [500]
        for k in kk:
                precision_list = get_precision_list( plays_full, plays_train,model,k=k)
                save_object(precision_list,precomputed_path+'precision_'+str(k)+'.pkl')

