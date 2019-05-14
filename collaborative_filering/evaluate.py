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
    print('hello')

    print('Loading data...')
    plays_full_path = precomputed_path + 'plays_full.pkl'
    norm_plays_full_path = precomputed_path + 'norm_plays_full.pkl'
    norm_plays_train_path = precomputed_path + 'norm_plays_train.pkl'
    model_path = precomputed_path + 'model.pkl'
 
    print('1/4')
    plays_full = read_object(plays_full_path).tocsr()
    print('2/4')
    norm_plays_full = read_object(norm_plays_full_path)
    print('3/4')
    norm_plays_train = read_object(norm_plays_train_path)
    model = read_object(model_path)
    print('4/4')

    return plays_full, norm_plays_full, norm_plays_train, model

def get_scores( plays_full, norm_plays_full, norm_plays_train,model, k=5):


        NUSERS,NARTISTS = plays_full.shape


        precision_list = []
        rnd_baseline_list = []
        upper_bound_list = []
        mrr_list = []
        ndcg_list = []
        completed = 0
        new_completed = 0
        for user_id in range(0,NUSERS):
                new_completed = (user_id +1)/ (NUSERS) * 100
                print_progress('Evaluating k=' + str(k) + '...  ', completed ,new_completed  )
                
                sr_rank = model.recommend(user_id, norm_plays_train,N=k ) 
                rnd_rank = np.round(np.random.rand(k))*(NARTISTS-1)

                scores = []
                relevants = []
                rnd_baseline_relevants = []
                upper_bound = 0
                x, nonzero = plays_full[user_id].nonzero()
                for artistid in nonzero:
                        upper_bound += (1 if plays_full[user_id, artistid] > 1 else 0)

                for artist_id, x in sr_rank:
                        ground_truth = plays_full[user_id,artist_id]
                        norm_ground_truth = norm_plays_full[user_id,artist_id]
                        
                        # if ground_truth != 0:
                        scores.append(norm_ground_truth)
                        relevants.append(1 if ground_truth > 1 else 0) 
                # rnd baseline
                for artist_id in rnd_rank:
                        ground_truth = plays_full[user_id,artist_id]
                        rnd_baseline_relevants.append(1 if ground_truth > 1 else 0) 
                


                ndcg_list.append(metrics.ndcg_at_k(scores, k))
                precision_list.append(sum(relevants)/k)
                mrr_list.append(metrics.mean_reciprocal_rank(relevants))
                rnd_baseline_list.append(metrics.precision_at_k(rnd_baseline_relevants, k))
                upper_bound_list.append(upper_bound/k)
         

        print(' ... completed', end='\n')
        return ndcg_list, precision_list, mrr_list,rnd_baseline_list, upper_bound_list

def evaluate(fakeDataset):

        precomputed_path =  './fake_precomputed_data/' if fakeDataset else './precomputed_data/'

        # load normalized data from pickle files
        plays_full,norm_plays_full, norm_plays_train,model = load_data(precomputed_path)

        kk = [10]
        for k in kk:
                ndcg_list, precision_list, mrr_list, rnd_baseline_list, upper_bound_list = get_scores( plays_full, norm_plays_full, norm_plays_train,model,k=k)
                
                save_object(ndcg_list,precomputed_path+'ndcg_list_'+str(k)+'.pkl')
                save_object(precision_list,precomputed_path+'precision_list_'+str(k)+'.pkl')
                save_object(rnd_baseline_list,precomputed_path+'rnd_baseline_list_'+str(k)+'.pkl')
                save_object(upper_bound_list,precomputed_path+'upper_bound_list_'+str(k)+'.pkl')
                save_object(mrr_list,precomputed_path+'mrr_list_'+str(k)+'.pkl')

