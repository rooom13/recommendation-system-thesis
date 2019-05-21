from ReadSave import *
import implicit
from content_based.TfidfRecommender import TfidfRecommender

from scipy.sparse import coo_matrix, csr_matrix
from numpy import array
import numpy as np
import sys
import time
import metrics as metricsf
import os
import pandas as pd



# Just for a prettier matrix print
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})


def print_progress(text, completed, new_completed):
     if (new_completed > completed): 
            completed = new_completed
            sys.stdout.write('\r'+text+ str(completed) + ' %' )
            sys.stdout.flush()



# load user_indices, artist_indices, plays 
def load_data(dataset_path,precomputed_path, methods):
    print('Evaluation started, loading data...', end='')
    plays_full_path = precomputed_path + 'plays_full.pkl'
    plays_train_path = precomputed_path + 'plays_train.pkl'
    norm_plays_full_path = precomputed_path + 'norm_plays_full.pkl'
    norm_plays_train_path = precomputed_path + 'norm_plays_train.pkl'
    artists_indices_path = precomputed_path + 'artist_index_index_artist.pkl'


    model_path = precomputed_path + 'model.pkl'
    bios_path = dataset_path + 'bios.txt'

 
    plays_full = read_object(plays_full_path).tocsr()
    print('1/7',end='...')
    plays_train = read_object(plays_train_path).tocsr()
    print('2/7', end='...')
    norm_plays_full = read_object(norm_plays_full_path).tocsr()
    print('3/7', end='...')
    norm_plays_train = read_object(norm_plays_train_path).tocsr()
    print('4/7', end='...')
    artist_index, index_artist = read_object(artists_indices_path)
    print('5/7', end='...')
    model = None if not methods['cf'] else read_object(model_path)

    print('5/7', end='...')
    ds = None if not methods['cb'] else pd.read_csv(bios_path,sep='\t') 
    print('7/7')

    return plays_full, plays_train, norm_plays_full, norm_plays_train, artist_index, index_artist, model, ds
#             rnd_baseline_list, upper_bound_list, precision_lists, ndcg_lists, mrr_lists, diversities  =  )

def get_scores(k,ds_bios, plays_full, plays_train, norm_plays_full, norm_plays_train,cf_model, cb_model,artist_index, index_artist, metrics, methods):

    NUSERS,NARTISTS = plays_full.shape

    rnd_baseline_list = []
    upper_bound_list = []
    
    cf_precision_list = []
    cb_precision_list = []
    cf_ndcg_list = []
    cb_ndcg_list = []
    cf_mrr_list = []
    cb_mrr_list = []
    cf_diversity = set()
    cb_diversity = set()



    msg2 = ('map ' if metrics['map'] else '') +\
    ('rnd ' if metrics['rnd'] else '') +\
    ('ub ' if metrics['ub'] else '') +\
    ('diversity ' if metrics['diversity'] else '') +\
    ('ndcg ' if metrics['ndcg'] else '') +\
    ('mrr ' if metrics['mrr'] else '') + ' -> ' +\
    ('CF ' if methods['cf'] else '') +\
    ('CB ' if methods['cb'] else '') 
    
    completed = 0
    new_completed = 0
    for user_id in [0]: #range(0,NUSERS):
        new_completed = (user_id +1)/ (NUSERS) * 100
        print_progress('Evaluating '+msg2+' k=' + str(k) + '\t  ', completed ,new_completed  )
        
        # Colaborative filtering rank
        cf_rank = None if not methods['cf'] else cf_model.recommend(user_id, norm_plays_train,N=k ) 


        # Content based rank 0/5
        # get artistid mapped to artistnames from user artist history
        user_history_indexs = (plays_train[user_id] > 1).nonzero()[1] 
        
        user_history =  None if not methods['cb'] else [index_artist[artistid] for artistid in user_history_indexs]
        # whichs indices in bios
        history_index_bios =  None if not methods['cb'] else ds_bios[ds_bios['id'].isin(user_history)].index.values
        # recommend
        rec_indices =  None if not methods['cb'] else cb_model.recommend_similars(history_index_bios,k)
        # which artists id are those indices 
        cb_rank =  None if not methods['cb'] else  [ds_bios.iloc[i]['id'] for i in rec_indices]
      

        # Random baseline rank
        rnd_rank = [] if not metrics['rnd'] else np.arange(NARTISTS +1)
        np.random.shuffle( rnd_rank ) 
        rnd_rank = rnd_rank.tolist() 
        rnd_rank =  rnd_rank[:k]        

        for i in user_history_indexs: 
            try:
               rnd_rank.remove(i)
            except:
                pass

        # print('rnd_rank',rnd_rank)

        rnd_relevants = []
        upper_bound = 0

        cf_ndcg_score = []
        cb_ndcg_score = []
        
        cf_relevants = []
        cb_relevants = []
        
        # rnd baseline
        if(metrics['rnd']):
                for artist_id in rnd_rank:
                        try:
                                ground_truth = plays_full[user_id,artist_id]
                        except:
                                ground_truth = 0
                        finally:
                                rnd_relevants.append(1 if ground_truth > 1 else 0) 
        
        # Upper bounds
        if(metrics['ub']):
                x, nonzero = plays_full[user_id].nonzero()
                for artist_id in nonzero:
                        ground_truth = plays_full[user_id,artist_id]
                        train = plays_train[user_id,artist_id]
                        if(train == 0 and ground_truth > 1):
                                upper_bound += 1

        
        # Diversities
        if(metrics['diversity']):
                cf_diversity.update([i for i,x in cf_rank])
                cb_diversity.update(cb_rank)

        # Relevants & ndcg score
        # print('cf_rank', cf_rank)
        # print('cb_rank', cb_rank)
        if(metrics['map'] or metrics['ndcg'] or metrics['mrr'] ):
                # Cf
                if(methods['cf']): 
                        for artist_id, x in cf_rank:
                                # relevants
                                ground_truth = plays_full[user_id,artist_id]
                                cf_relevants.append(1 if ground_truth > 1 else 0) 
                                # ndcg
                                norm_ground_truth = norm_plays_full[user_id,artist_id]
                                cf_ndcg_score.append(norm_ground_truth)
                        
                # Cb 
                if(methods['cb']): 
                        for artist in cb_rank:
                                try:
                                        artist_id = artist_index[artist]
                                except KeyError:
                                        continue
                                # relevants
                                ground_truth = plays_full[user_id,artist_id]
                                cb_relevants.append(1 if ground_truth > 1 else 0) 
                                # ndcg
                                norm_ground_truth = norm_plays_full[user_id,artist_id]
                                cb_ndcg_score.append(norm_ground_truth)
        

        # rnd baseline & upper bound
        rnd_baseline_list.append(sum(rnd_relevants)/k)
        upper_bound_list.append( 1 if upper_bound/k > 1 else upper_bound/k)
        # nDCG
        cf_ndcg_list.append(metricsf.ndcg_at_k(cf_ndcg_score, k))
        cb_ndcg_list.append(metricsf.ndcg_at_k(cb_ndcg_score, k))
        # precision
        cf_precision_list.append(sum(cf_relevants)/k)
        cb_precision_list.append(sum(cb_relevants)/k)
        # mrr
        cf_mrr_list.append(metricsf.mean_reciprocal_rank(cf_relevants))
        cb_mrr_list.append(metricsf.mean_reciprocal_rank(cb_relevants))

            
        

    print(' ...completed', end='\n')
    # return ndcg_list, precision_list, mrr_list,rnd_baseline_list, upper_bound_list, diversity
    return rnd_baseline_list, upper_bound_list, (cf_precision_list, cb_precision_list), (cf_ndcg_list, cb_ndcg_list), (cf_mrr_list, cb_mrr_list) , (cf_diversity, cb_diversity)  
        
def get_paths(dataset_path, results_path):
    precomputed_path =  dataset_path + 'precomputed_data/'  

    if not os.path.exists(results_path):
            os.mkdir(results_path)

    cf_results_path = results_path + 'collaborating_filtering/' 
    cb_results_path = results_path + 'content_based/' 
    
    if not os.path.exists(cf_results_path):
            os.mkdir(cf_results_path)

    if not os.path.exists(cb_results_path):
            os.mkdir(cb_results_path)
    return precomputed_path, [cf_results_path, cb_results_path]
    
DEFAULT_METRICS = {
    'map': False, 
    'diversity': True, 
    'ndcg': False, 
    'mrr': False,
    'rnd': True,
    'ub': True,
    }
DEFAULT_METHODS= {
    'cf': True,
    'cb': True
}    
def evaluate(dataset_path,results_path,kk=[10,100,200], metrics= DEFAULT_METRICS, methods=DEFAULT_METHODS):
    precomputed_path, results_paths = get_paths(dataset_path, results_path)
       
                
    # load normalized data from pickle files
    plays_full, plays_train, norm_plays_full, norm_plays_train, artist_index, index_artist, cf_model, ds_bios = load_data(dataset_path,precomputed_path, methods)

    
    cb_model = None if not methods['cb'] else TfidfRecommender(ds_bios['bio'].tolist())


    for k in kk:
            
        rnd_baseline_list, upper_bound_list, precision_lists, ndcg_lists, mrr_lists, diversities  = get_scores(k,ds_bios, plays_full, plays_train, norm_plays_full, norm_plays_train,cf_model, cb_model,artist_index, index_artist,metrics, methods)

        if(metrics['rnd']):
            save_object(rnd_baseline_list,results_paths[0]+'rnd_baseline_list_'+str(k)+'.pkl')
        if(metrics['ub']):
            save_object(upper_bound_list,results_paths[0]+'upper_bound_list_'+str(k)+'.pkl')
            
        for i in range(0,len(results_paths)):
                if(i == 0 and not methods['cf'] ):
                        continue
                elif(i == 1 and not methods['cb'] ):
                        continue
                results_path = results_paths[i]
                if(metrics['ndcg']):

                        save_object(ndcg_lists[i],results_path+'ndcg_list_'+str(k)+'.pkl')
                if(metrics['map']):
                        
                        save_object(precision_lists[i],results_path+'precision_list_'+str(k)+'.pkl')
                if(metrics['mrr']):
                        
                        save_object(mrr_lists[i],results_path+'mrr_list_'+str(k)+'.pkl')
                if(metrics['diversity']):                
                        save_object(diversities[i],results_path+'diversity_'+str(k)+'.pkl')

