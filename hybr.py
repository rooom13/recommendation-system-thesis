import pandas as pd
from ReadSave import *
import metrics
import atexit

import numpy as np
from content_based.TfidfRecommender import TfidfRecommender

import sys
import os

# Just for a prettier matrix print
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def print_progress(completed, new_completed, total,p):
     if (new_completed > completed): 
            completed = new_completed
            percentage = round(new_completed/total *100,6)
            sys.stdout.write('\rusers: '+ str(completed+1) + '/' + str(total) +' - '+ str(percentage)+' map_k10: ' + str(sum(p['cf'][10][:10])/10) )
            sys.stdout.flush()



# load user_indices, artist_indices, plays 
def load_data(dataset_path,precomputed_path):
    print('Evaluation started, loading data...', end='')
    plays_full_path = precomputed_path + 'plays_full.pkl'
    plays_train_path = precomputed_path + 'plays_train.pkl'
    norm_plays_full_path = precomputed_path + 'norm_plays_full.pkl'
    norm_plays_train_path = precomputed_path + 'norm_plays_train.pkl'
    artists_indices_path = precomputed_path + 'artist_index_index_artist.pkl'


    model_path = precomputed_path + 'model.pkl'
    bios_path = dataset_path + 'bios.txt'

#     print('0/7')
    plays_full = read_object(plays_full_path).tocsr()
#     print('1/7')
    plays_train = read_object(plays_train_path).tocsr()
#     print('2/7')
    norm_plays_full = read_object(norm_plays_full_path).tocsr()
#     print('3/7')
    norm_plays_train = read_object(norm_plays_train_path).tocsr()
#     print('4/7')
    artist_index, index_artist = read_object(artists_indices_path)
#     print('5/7')
    model = read_object(model_path)
#     print('5/7')
    ds = pd.read_csv(bios_path,sep='\t') 
#     print('7/7')

    return plays_full, plays_train, norm_plays_full, norm_plays_train, artist_index, index_artist, model, ds


def readme():
        print('Loading backup!')
        (the_user_id, rnd_baselines, upper_bounds, diversities, precisions, mrrs, ndcgs) = read_object('backup.pkl')
        print('Starting from User', the_user_id)
        return the_user_id, rnd_baselines, upper_bounds, diversities, precisions, mrrs, ndcgs

loadBackup = False
saveBackup = False


precisions = {
        'hb': {
                5: [],
                10: [],
                100: [],
                200: [],
                500: []
        },
        'cb': {
                5: [],
                10: [],
                100: [],
                200: [],
                500: []
        },
        'cf': {
                5: [],
                10: [],
                100: [],
                200: [],
                500: []
        }
}
mrrs = {
        'hb': {
                5: [],
                10: [],
                100: [],
                200: [],
                500: []
        },
        'cf': {
                5: [],
                10: [],
                100: [],
                200: [],
                500: []
        },
        'cb': {
                5: [],
                10: [],
                100: [],
                200: [],
                500: []
        }
}
ndcgs = {
        'hb': {
                5: [],
                10: [],
                100: [],
                200: [],
                500: []
        },
        'cb': {
                5: [],
                10: [],
                100: [],
                200: [],
                500: []
        },
        'cf': {
                5: [],
                10: [],
                100: [],
                200: [],
                500: []
        }
}

diversities = {
        'hb': {
                5: set(),
                10: set(),
                100: set(),
                200: set(),
                500: set()
        },
        'cb': {
                5: set(),
                10: set(),
                100: set(),
                200: set(),
                500: set()
        },
        'cf': {
                5: set(),
                10: set(),
                100: set(),
                200: set(),
                500: set()
        }
}

rnd_baselines = {
        5: [],
        10: [],
        100: [],
        200: [],
        500: []
}
upper_bounds = {
        5: [],
        10: [],
        100: [],
        200: [],
        500: []
}

the_user_id = 0
if loadBackup:
        the_user_id, rnd_baselines, upper_bounds, diversities, precisions, mrrs, ndcgs = readme()



def get_cb_rank(ds_bios, user_history, cb_model,k):
        history_index_bios = ds_bios[ds_bios['id'].isin(user_history)].index.values
        rec_indices = cb_model.recommend_similars(history_index_bios,k)
        return [ds_bios.iloc[i]['id'] for i in rec_indices]

def get_rnd_rank(NARTISTS,exclude,k):
        rnd_rank = np.arange(NARTISTS)
        np.random.shuffle( rnd_rank ) 
        rnd_rank = rnd_rank.tolist() 
        rnd_rank =  rnd_rank[:k]        
        # exclude already listened
        for i in exclude: 
            try:
               rnd_rank.remove(i)
            except:
                pass
        return rnd_rank


def mix(cf_rank,cb_rank,artist_index):
        hybrid = []

        for i in range(0,len(cb_rank)):
                try: 
                        cf_id, x = cf_rank[i]
                        if not cf_id in hybrid:
                                hybrid.append(cf_id)
                except:
                        pass
                try:
                        cb_id = artist_index[cb_rank[i]]
                        if not cb_id in hybrid:
                                hybrid.append(cb_id)
                except: 
                        pass
        return hybrid

def get_scores(ds_bios, plays_full, plays_train, norm_plays_full, norm_plays_train,cf_model, cb_model,artist_index, index_artist):
    
    NUSERS,NARTISTS = plays_full.shape    

    global the_user_id
   

    completed = 0
    new_completed = 0


    lightUsers = get_rnd_rank(NUSERS,[],1000)

    for user_id in lightUsers: #range(the_user_id,NUSERS):
        the_user_id = user_id
        print_progress( completed,user_id,NUSERS,precisions)

        # Colaborative filtering rank
        cf_rank =cf_model.recommend(user_id, norm_plays_train,N=500 ) 

        # get history of artistid
        user_history_indexs = (plays_train[user_id] > 1).nonzero()[1] 

        # mapped to artistnames from user artist history
        user_history =  [index_artist[artistid] for artistid in user_history_indexs]
        
        # Content based rank 
        cb_rank = get_cb_rank(ds_bios, user_history, cb_model,500)
      
        # Hybrid mixed rank
        hb_rank = mix(cf_rank, cb_rank, artist_index)[:500]

        # Random baseline rank
        rnd_rank = get_rnd_rank(NARTISTS,user_history_indexs, 500)
        
        hb_scores = []
        cb_scores = []
        cf_scores = []
        
        hb_relevants = []
        cb_relevants = []
        cf_relevants = []

        rnd_relevants = []
        upper_bound = 0
        
        # Hybrid
        for artist_id in hb_rank:     
                ground_truth = plays_full[user_id,artist_id]
                hb_relevants.append(1 if ground_truth > 1 else 0)                 
                norm_ground_truth = norm_plays_full[user_id,artist_id]
                hb_scores.append(norm_ground_truth)
        
        # Collaborative Filtering
        for artist_id, x in cf_rank:     
                ground_truth = plays_full[user_id,artist_id]
                cf_relevants.append(1 if ground_truth > 1 else 0)                 
                norm_ground_truth = norm_plays_full[user_id,artist_id]
                cf_scores.append(norm_ground_truth)
       
        # Rnd Baseline
        for artist_id in rnd_rank:     
                try:
                        ground_truth = plays_full[user_id,artist_id]
                except:
                        ground_truth = 0
                finally:
                        rnd_relevants.append(1 if ground_truth > 1 else 0) 
        
        # Upper Bound
        x, nonzero = plays_full[user_id].nonzero()
        for artist_id in nonzero:
                ground_truth = plays_full[user_id,artist_id]
                try:
                        train = plays_train[user_id,artist_id]
                except:
                        train = 0
                finally:
                        if(train == 0 and ground_truth > 1):
                                upper_bound += 1

        # Content based
        for artist_name in cb_rank:     
                try:
                        artist_id = artist_index[artist_name]
                except KeyError:
                        continue

                ground_truth = plays_full[user_id,artist_id]
                cb_relevants.append(1 if ground_truth > 1 else 0)                 
                norm_ground_truth = norm_plays_full[user_id,artist_id]
                cb_scores.append(norm_ground_truth)

        # ks
        for k in [5,10,100,200,500]:

                rnd_baselines[k].append(sum(rnd_relevants[:k])/k)
                upper_bounds[k].append(1 if upper_bound/k > 1 else upper_bound/k)

                diversities['hb'][k].update(hb_rank[:k])
                precisions['hb'][k].append(sum(hb_relevants[:k])/k)
                mrrs['hb'][k].append(metrics.reciprocal_rank(hb_relevants[:k]))
                ndcgs['hb'][k].append(metrics.ndcg_at_k(hb_scores[:k], k))
                
                diversities['cb'][k].update(cb_rank[:k])
                precisions['cb'][k].append(sum(cb_relevants[:k])/k)
                mrrs['cb'][k].append(metrics.reciprocal_rank(cb_relevants[:k]))
                ndcgs['cb'][k].append(metrics.ndcg_at_k(cb_scores[:k], k))
                
                diversities['cf'][k].update([i for i,x in cf_rank[:k]])
                precisions['cf'][k].append(sum(cf_relevants[:k])/k)
                mrrs['cf'][k].append(metrics.reciprocal_rank(cf_relevants[:k]))
                ndcgs['cf'][k].append(metrics.ndcg_at_k(cf_scores[:k], k))


    return rnd_baselines, upper_bounds, diversities, precisions, mrrs, ndcgs



@atexit.register
def saveme():
        if(saveBackup):
                print('saving backup!')
                save_object((the_user_id, rnd_baselines, upper_bounds, diversities, precisions, mrrs, ndcgs),'backup.pkl')
                print('saved')

methods = {
        'cb': True,
        'cf': True,
        'hb': True
}

def evaluate(dataset_path, results_path, kk=[5,10,100,200,500], metrics={}, methods={}):


    precomputed_path = dataset_path + 'precomputed_data/'

    plays_full, plays_train, norm_plays_full, norm_plays_train, artist_index, index_artist, cf_model, ds_bios = load_data(dataset_path,precomputed_path)

    cb_model = TfidfRecommender(ds_bios['bio'].tolist())

    rnd_baselines, upper_bounds, diversities, precisions, mrrs, ndcgs  = get_scores(ds_bios, plays_full, plays_train, norm_plays_full, norm_plays_train,cf_model, cb_model,artist_index, index_artist)

    results_path_hybrid = results_path + 'hybrid/'
    results_path_cf = results_path + 'collaborative_filtering/'
    results_path_cb = results_path + 'content_based/'
    




    result_paths = results_path_hybrid
    for k in kk:
            save_object(diversities['hb'][k],result_paths+'diversity_'+str(k)+'.pkl')
            save_object(precisions['hb'][k],result_paths+'precision_list_'+str(k)+'.pkl')
            save_object(ndcgs['hb'][k],result_paths+'ndcg_list_'+str(k)+'.pkl')
            save_object(mrrs['hb'][k],result_paths+'mrr_list_'+str(k)+'.pkl')
    
    result_paths = results_path_cb

    for k in kk:
            save_object(diversities['cb'][k],result_paths+'diversity_'+str(k)+'.pkl')
            save_object(precisions['cb'][k],result_paths+'precision_list_'+str(k)+'.pkl')
            save_object(ndcgs['cb'][k],result_paths+'ndcg_list_'+str(k)+'.pkl')
            save_object(mrrs['cb'][k],result_paths+'mrr_list_'+str(k)+'.pkl')
    
    result_paths = results_path_cf
        
    for k in kk:

            save_object(rnd_baselines[k],result_paths+'rnd_baseline_list_'+str(k)+'.pkl')
            save_object(upper_bounds[k],result_paths+'upper_bound_list_'+str(k)+'.pkl')
            save_object(diversities['cf'][k],result_paths+'diversity_'+str(k)+'.pkl')
            save_object(precisions['cf'][k],result_paths+'precision_list_'+str(k)+'.pkl')
            save_object(ndcgs['cf'][k],result_paths+'ndcg_list_'+str(k)+'.pkl')
            save_object(mrrs['cf'][k],result_paths+'mrr_list_'+str(k)+'.pkl')

    
# fakeDataset = True
# dataset_path= './fake_dataset/' if fakeDataset else './dataset/'
# results_path= './fake_results/' if fakeDataset else './results/'
 
# evaluate(dataset_path, results_path)

