from ReadSave import *
import implicit
from scipy.sparse import coo_matrix, csr_matrix
from numpy import array
import numpy as np
import sys
import time
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
     
    plays_full_path = precomputed_path + 'norm_plays_full.pkl'
    plays_train_path = precomputed_path + 'norm_plays_train.pkl'
    model_path = precomputed_path + 'model.pkl'
 
    plays_full = read_object(plays_full_path)
    plays_train = read_object(plays_train_path)
    model = read_object(model_path)

    return plays_full, plays_train, model

def get_NDCG_list( plays_full, plays_train,model, k=5):


        NUSERS,NARTISTS = plays_full.shape


        ndcg_list = []
        completed = 0
        new_completed = 0
        for user_id in range(0,NUSERS):
                new_completed = (user_id +1)/ (NUSERS) * 100
                
                print_progress('Evaluating NDCG k=' + str(k) + '...  ', completed ,new_completed  )
                sr_rank = model.recommend(user_id, plays_train,N=k ) 
                scores = []
                for artist_id, x in sr_rank:
                        ground_truth = plays_full[user_id,artist_id]
                        if ground_truth != 0:
                                scores.append(ground_truth)
                ndcg_list.append(metrics.ndcg_at_k(scores, k))
        print(' ... completed', end='\n')
        return ndcg_list

def evaluate_ndcg(fakeDataset):
        precomputed_path =  './fake_precomputed_data/' if fakeDataset else './precomputed_data/'

        # load normalized data from pickle files
        plays_full, plays_train,model = load_data(precomputed_path)

        kk = [10,100,500]
        for k in kk:
                NDCG_list = get_NDCG_list( plays_full, plays_train,model,k=k)
                save_object(NDCG_list,precomputed_path+'ndcg_'+str(k)+'.pkl')


