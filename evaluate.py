from ReadSave import *
import metrics
import atexit

import numpy as np
import pandas as pd
import sys
import os


## UTILS
def print_progress(completed, new_completed, total):
     if (new_completed > completed): 
            completed = new_completed
            percentage = round(new_completed/total *100,6)
            sys.stdout.write('\rusers: '+ str(completed+1) + '/' + str(total) +' - '+ str(percentage))
            sys.stdout.flush()

# Load previous saved execution backup
def load_backup():
        print('Loading backup!')
        (the_user_id, rnd_baselines, upper_bounds, diversities, precisions, mrrs, ndcgs) = read_object('backup.pkl')
        print('Starting from User', the_user_id)
        return the_user_id, rnd_baselines, upper_bounds, diversities, precisions, mrrs, ndcgs

# In case of error or program exit (ex ctrl + C)
@atexit.register
def saveme():
        if(saveBackup):
                print('saving backup!')
                save_object((the_user_id, rnd_baselines, upper_bounds, diversities, precisions, mrrs, ndcgs),'backup.pkl')
                print('saved')

# till yet this needs to be manually set
loadBackup = False
saveBackup = False

# creates dir if path doesn't exist 
def mkdir_ifNot_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)   

# load user_indices, artist_indices, plays 
def load_data(dataset_path,precomputed_path, methods):
    print('Evaluation started, cahing data...', end='')
    plays_full_path = precomputed_path + 'plays_full.pkl'
    plays_train_path = precomputed_path + 'plays_train.pkl'
    norm_plays_full_path = precomputed_path + 'norm_plays_full.pkl'
    norm_plays_train_path = precomputed_path + 'norm_plays_train.pkl'
    artists_indices_path = precomputed_path + 'artist_index_index_artist.pkl'
    cf_model_path = precomputed_path + 'cf_model.pkl'
    tfIdfRecommender_path = precomputed_path + 'tfIdfRecommender.pkl'
    bios_path = dataset_path + 'bios.txt'

    print('this may take some minutes...')
    plays_full = read_object(plays_full_path).tocsr()
    plays_train = read_object(plays_train_path).tocsr()
    norm_plays_full = read_object(norm_plays_full_path).tocsr()
    norm_plays_train = read_object(norm_plays_train_path).tocsr()

    print('..almost..')
    artist_index, index_artist, cb_model, ds, cf_model = None, None, None, None, None

    if methods['hb'] or methods['cb']: 
        artist_index, index_artist = read_object(artists_indices_path)
        cb_model = read_object(tfIdfRecommender_path)
        ds = pd.read_csv(bios_path,sep='\t')

    print('5/6')
    if methods['hb'] or methods['cf']: 
        cf_model = read_object(cf_model_path)
    print('6/6')

    return plays_full, plays_train, norm_plays_full, norm_plays_train, artist_index, index_artist, cf_model, cb_model, ds



# Global variables for easy backup 
precisions = {'hb': {5: [],10: [],100: [],200: [],500: []},'cb': {5: [],10: [],100: [],200: [],500: []},'cf': {5: [],10: [],100: [],200: [],500: []}}
mrrs = {'hb': {5: [],10: [],100: [],200: [],500: []},'cf': {5: [],10: [],100: [],200: [],500: []},'cb': {5: [],10: [],100: [],200: [],500: []}}
ndcgs = {'hb': {5: [],10: [],100: [],200: [],500: []},'cb': {5: [],10: [],100: [],200: [],500: []},'cf': {5: [],10: [],100: [],200: [],500: []}}
diversities = {'hb': {5: set(),10: set(),100: set(),200: set(),500: set()},'cb': {5: set(),10: set(),100: set(),200: set(),500: set()},'cf': {5: set(),10: set(),100: set(),200: set(),500: set()}}
rnd_baselines = {5: [],10: [],100: [],200: [],500: []}
upper_bounds = {5: [],10: [],100: [],200: [],500: []}

the_user_id = 0
if loadBackup:
        the_user_id, rnd_baselines, upper_bounds, diversities, precisions, mrrs, ndcgs = load_backup()

def get_cb_rank(ds_bios, user_history, cb_model,artist_index, k):
        history_index_bios = ds_bios[ds_bios['id'].isin(user_history)].index.values
        rec_indices = cb_model.recommend_similars(history_index_bios,k)
        names = [ds_bios.iloc[i]['id'] for i in rec_indices]
        nonCF_artistid = -1
        cb_rank = []
        for name in names:
                try:
                        cf_id = artist_index[name]
                except:
                        cf_id = nonCF_artistid
                        nonCF_artistid = nonCF_artistid-1
                finally:
                        cb_rank.append(cf_id) 
        return cb_rank


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


def mix(cf_rank,cb_rank):
        hybrid = []
        for i in range(0,len(cb_rank)):
                try: 
                        cf_id = cf_rank[i]
                        if not cf_id in hybrid:
                                hybrid.append(cf_id)
                except:
                        pass
                cb_id = cb_rank[i]
                if not cb_id in hybrid:
                        hybrid.append(cb_id)
        return hybrid

def get_scores(ds_bios, plays_full, plays_train, norm_plays_full, norm_plays_train,cf_model, tfIdfRecommender,artist_index, index_artist, methodKeys,kk):
    
    NUSERS,NARTISTS = plays_full.shape    

    global the_user_id
   
    completed = 0
    new_completed = 0

    lightUsers = get_rnd_rank(NUSERS,[],100)

    ranks = {}

    for user_id in lightUsers: #range(the_user_id,NUSERS):
        the_user_id = user_id
        print_progress(completed, user_id, NUSERS)

        # Colaborative filtering rank
        ranks['cf'] =[i for i,x in cf_model.recommend(user_id, norm_plays_train,N=max(kk) ) ]
        # get history of artistid
        user_history_indexs = (plays_train[user_id] > 1).nonzero()[1] 

        # mapped to artistnames from user artist history
        user_history =  [index_artist[artistid] for artistid in user_history_indexs]
        
        # Content based rank 
        ranks['cb'] = get_cb_rank(ds_bios, user_history, tfIdfRecommender, artist_index,max(kk))

        # Hybrid mixed rank
        ranks['hb'] = mix(ranks['cf'], ranks['cb'])[:max(kk)]

        # Random baseline rank
        rnd_rank = get_rnd_rank(NARTISTS,user_history_indexs, max(kk))

        scores = {}
        relevants={}
        rnd_relevants = []
        upper_bound = 0

        # Calculate relevants and scores for each method
        for method in methodKeys:
                scores[method] = []
                relevants[method] = []
                for artist_id in ranks[method]:
                        ground_truth = plays_full[user_id,artist_id]
                        relevants[method].append(1 if ground_truth > 1 else 0)
                        
                        norm_ground_truth = norm_plays_full[user_id,artist_id]
                        scores[method].append(norm_ground_truth)
       
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

        # save user metrics
        for k in kk:
                rnd_baselines[k].append(sum(rnd_relevants[:k])/k)
                upper_bounds[k].append(1 if upper_bound/k > 1 else upper_bound/k)

        for method in methodKeys:
            for k in kk:
                diversities[method][k].update(ranks[method][:k])
                precisions[method][k].append(sum(relevants[method][:k])/k)
                ndcgs[method][k].append(metrics.ndcg_at_k(scores[method][:k], k))
                mrrs[method][k].append(metrics.reciprocal_rank(relevants[method][:k]))

    return rnd_baselines, upper_bounds, diversities, precisions, mrrs, ndcgs


def saveMetrics(kk,methods,result_paths, diversities, precisions, ndcgs, mrrs):
   
    for method, result_paths in zip(methods, result_paths):
        mkdir_ifNot_exist(result_paths)
        for k in kk:
            save_object(diversities[method][k],result_paths+'diversity_'+str(k)+'.pkl')
            save_object(precisions[method][k],result_paths+'precision_list_'+str(k)+'.pkl')
            save_object(ndcgs[method][k],result_paths+'ndcg_list_'+str(k)+'.pkl')
            save_object(mrrs[method][k],result_paths+'mrr_list_'+str(k)+'.pkl')

def evaluate(dataset_path, results_path, kk=[5,10,100,200,500], metrics={}, methods={}):

    precomputed_path = dataset_path + 'precomputed_data/'
    # which methods to evaluate
    methodsKeys = []
    result_paths = []
    if methods['cb']:
            methodsKeys.append('cb')
            result_paths.append(results_path + 'content_based/')
    if methods['cf']:
            methodsKeys.append('cf')
            result_paths.append(results_path + 'collaborative_filtering/')
    if methods['hb']:
            methodsKeys.append('hb')
            result_paths.append(results_path + 'hybrid/')

    # Load user-item matrixes, models, indexes...
    plays_full, plays_train, norm_plays_full, norm_plays_train, artist_index, index_artist, cf_model, cb_model, ds_bios = load_data(dataset_path,precomputed_path, methods)

    # evaluate
    rnd_baselines, upper_bounds, diversities, precisions, mrrs, ndcgs  = get_scores(ds_bios, plays_full, plays_train, norm_plays_full, norm_plays_train,cf_model, cb_model,artist_index, index_artist, methodsKeys, kk)
    
    # Save metrics results
    mkdir_ifNot_exist(results_path)
    saveMetrics(kk, methodsKeys, result_paths, diversities, precisions, ndcgs, mrrs)

    # Save rnd baseline & upper bound
    for precisions, result_path in zip([rnd_baselines,upper_bounds],[results_path + 'rnd_baseline/', results_path + 'upper_bound/']):
        for k in kk:
            mkdir_ifNot_exist(result_path)
            save_object(precisions[k],result_path+'precision_list_'+str(k)+'.pkl')