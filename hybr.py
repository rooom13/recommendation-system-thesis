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
            sys.stdout.write('\rusers: '+ str(completed+1) + '/' + str(total) +' - '+ str(percentage)+' map_k10: ' + str(sum(p[10][:10])/10) )
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

    print('0/7')
    plays_full = read_object(plays_full_path).tocsr()
    print('1/7')
    plays_train = read_object(plays_train_path).tocsr()
    print('2/7')
    norm_plays_full = read_object(norm_plays_full_path).tocsr()
    print('3/7')
    norm_plays_train = read_object(norm_plays_train_path).tocsr()
    print('4/7')
    artist_index, index_artist = read_object(artists_indices_path)
    print('5/7')
    model = read_object(model_path)
    print('5/7')
    ds = pd.read_csv(bios_path,sep='\t') 
    print('7/7')

    return plays_full, plays_train, norm_plays_full, norm_plays_train, artist_index, index_artist, model, ds


def readme():
        print('Loading backup!')
        (the_user_id, diversities, precisions, mrrs, ndcgs) = read_object('backup.pkl')
        print('Starting from User', the_user_id)
        return the_user_id, diversities, precisions, mrrs, ndcgs

loadBackup = False
saveBackup = False


precisions = {
        5: [],
        10: [],
        100: [],
        200: [],
        500: []
}
mrrs = {
        5: [],
        10: [],
        100: [],
        200: [],
        500: []
}
ndcgs = {
        5: [],
        10: [],
        100: [],
        200: [],
        500: []
}

diversities = {
        5: set(),
        10: set(),
        100: set(),
        200: set(),
        500: set()
}
the_user_id = 0
if loadBackup:
        the_user_id, diversities, precisions, mrrs, ndcgs = readme()



def get_cb_rank(ds_bios, user_history, cb_model,k):
        history_index_bios = ds_bios[ds_bios['id'].isin(user_history)].index.values
        rec_indices = cb_model.recommend_similars(history_index_bios,k)
        return [ds_bios.iloc[i]['id'] for i in rec_indices]


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

    for user_id in range(the_user_id,NUSERS):
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
      


        hybrid_rank = mix(cf_rank, cb_rank, artist_index)
        


        sys.exit()

        scores = []
        relevants = []

        for artist_id in hybrid_rank:
               
                ground_truth = plays_full[user_id,artist_id]
                # print(ground_truth) 
                relevants.append(1 if ground_truth > 1 else 0) 
                
                norm_ground_truth = norm_plays_full[user_id,artist_id]
                scores.append(norm_ground_truth)

        

        # ks
        for k in [5,10,100,200,500]:
                diversities[k].update(rec_indices[:k])
                precisions[k].append(sum(relevants[:k])/k)
                mrrs[k].append(metrics.mean_reciprocal_rank(relevants[:k]))
                ndcgs[k].append(metrics.ndcg_at_k(scores[:k], k))

    return diversities, precisions, mrrs, ndcgs



@atexit.register
def saveme():
        if(saveBackup):
                print('saving backup!')
                save_object((the_user_id, diversities, precisions, mrrs, ndcgs),'backup.pkl')
                print('saved')

def evaluate(dataset_path, results_path):


    precomputed_path = dataset_path + 'precomputed_data/'

    plays_full, plays_train, norm_plays_full, norm_plays_train, artist_index, index_artist, cf_model, ds_bios = load_data(dataset_path,precomputed_path)

    cb_model = TfidfRecommender(ds_bios['bio'].tolist())

    diversities, precisions, mrrs, ndcgs  = get_scores(ds_bios, plays_full, plays_train, norm_plays_full, norm_plays_train,cf_model, cb_model,artist_index, index_artist)

    results_path = results_path + 'hybrid/'
    for k in [5,10,100,200,500]:
            save_object(diversities[k],results_path+'diversity_'+str(k)+'.pkl')
            save_object(precisions[k],results_path+'precision_list_'+str(k)+'.pkl')
            save_object(ndcgs[k],results_path+'ndcg_list_'+str(k)+'.pkl')
            save_object(mrrs[k],results_path+'mrr_list_'+str(k)+'.pkl')

    
fakeDataset = True
dataset_path= './fake_dataset/' if fakeDataset else './dataset/'
results_path= './fake_results/' if fakeDataset else './results/'
 
evaluate(dataset_path, results_path)

