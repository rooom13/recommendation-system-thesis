from ReadSave import *
import implicit
from scipy.sparse import coo_matrix, csr_matrix
from numpy import array
import numpy as np
import sys
import time
import metrics
import os


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
    print('Collaborative filtering:\t - Evaluation started, loading data...', end='')
    plays_full_path = precomputed_path + 'plays_full.pkl'
    norm_plays_full_path = precomputed_path + 'norm_plays_full.pkl'
    norm_plays_train_path = precomputed_path + 'norm_plays_train.pkl'
    model_path = precomputed_path + 'model.pkl'
 
    print('1/4',end='...')
    plays_full = read_object(plays_full_path).tocsr()
    print('2/4', end='...')
    norm_plays_full = read_object(norm_plays_full_path)
    print('3/4', end='...')
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

        diversity = set()


        completed = 0
        new_completed = 0
        for user_id in range(0,NUSERS):
                new_completed = (user_id +1)/ (NUSERS) * 100
                print_progress('\tEvaluating  CF SR k=' + str(k) + '\t  ', completed ,new_completed  )
                
                sr_rank = model.recommend(user_id, norm_plays_train,N=k ) 
                
                diversity.update([i for i,x in sr_rank])
                
                
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
                
                upper_bound_list.append( 1 if upper_bound/k > 1 else upper_bound/k)
         

        print(' ...completed', end='\n')
        return ndcg_list, precision_list, mrr_list,rnd_baseline_list, upper_bound_list, diversity

def evaluate(datasetPath,results_path,kk=[10,100,200]):

        precomputed_path =  datasetPath + 'precomputed_data/'

        if not os.path.exists(results_path):
                os.mkdir(results_path)

        results_path = results_path + 'collaborating_filtering/' 
        
        if not os.path.exists(results_path):
                os.mkdir(results_path)
                

        # load normalized data from pickle files
        plays_full,norm_plays_full, norm_plays_train,model = load_data(precomputed_path)



        for k in kk:
                ndcg_list, precision_list, mrr_list, rnd_baseline_list, upper_bound_list, diversity = get_scores( plays_full, norm_plays_full, norm_plays_train,model,k=k)
                print('diversity',len(diversity))
                save_object(ndcg_list,results_path+'ndcg_list_'+str(k)+'.pkl')
                save_object(precision_list,results_path+'precision_list_'+str(k)+'.pkl')
                save_object(rnd_baseline_list,results_path+'rnd_baseline_list_'+str(k)+'.pkl')
                save_object(upper_bound_list,results_path+'upper_bound_list_'+str(k)+'.pkl')
                save_object(mrr_list,results_path+'mrr_list_'+str(k)+'.pkl')
                save_object(diversity,results_path+'diversity_'+str(k)+'.pkl')

