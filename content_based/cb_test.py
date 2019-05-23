import pandas as pd
from ReadSave import *
import metrics
import atexit

import numpy as np
from TfidfRecommender import TfidfRecommender

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

def loadData(dataset_path):
    print('Content based:\t - Evaluation started, loading data...', end='')
    precomputed_path =  dataset_path + 'precomputed_data/' 
    plays_full_path = precomputed_path + 'plays_full.pkl'
    plays_train_path = precomputed_path + 'plays_train.pkl'
    norm_plays_full_path = precomputed_path + 'norm_plays_full.pkl'
    artists_indices_path = precomputed_path + 'artist_index_index_artist.pkl'
    bios_path = dataset_path + 'bios.txt'

 
    plays_full = read_object(plays_full_path).tocsr()
    print('1/5')
    plays_train = read_object(plays_train_path).tocsr()
    print('2/5')
    norm_plays_full = read_object(norm_plays_full_path).tocsr()
    print('3/5')
    ds = pd.read_csv(bios_path,sep='\t') 
    print('4/5')
    artist_index, index_artist = read_object(artists_indices_path)
    print('5/5')



    return plays_full, plays_train, norm_plays_full,ds, artist_index, index_artist

def readme():
        print('Loading backup!')
        (the_user_id, diversities, precisions, mrrs, ndcgs) = read_object('backup.pkl')
        print('Starting from User', the_user_id)
        return the_user_id, diversities, precisions, mrrs, ndcgs

loadBackup = True


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
     
def get_scores(ds_bios,plays_full,plays_train,norm_plays_full, model,artist_index, index_artist):
    
    NUSERS,NARTISTS = plays_full.shape    

    global the_user_id
   

    completed = 0
    new_completed = 0

    for user_id in range(the_user_id,NUSERS):
        the_user_id = user_id
        print_progress( completed,user_id,NUSERS,precisions)

        # user id mapped to usernames
        user_history = [index_artist[artistid] for artistid in (plays_train[user_id] > 1).nonzero()[1] ]

        # whichs indices in abios
        history_index_bios = ds_bios[ds_bios['id'].isin(user_history)].index.values
        # recommend

        rec_indices = model.recommend_similars(history_index_bios,500)

        # which artists id are those indices 
        rec_artists = [ds_bios.iloc[i]['id'] for i in rec_indices]
        

        scores = []
        relevants = []

        for artist in rec_artists:
                try:
                        artist_id = artist_index[artist]

                except KeyError:
                        continue
                
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
        print('saving backup!')
        save_object((the_user_id, diversities, precisions, mrrs, ndcgs),'backup.pkl')
        print('saved')

def evaluate(dataset_path):

    plays_full, plays_train, norm_plays_full, ds_bios, artist_index, index_artist = loadData(dataset_path)

    model = TfidfRecommender(ds_bios['bio'].tolist())

    diversities, precisions, mrrs, ndcgs = get_scores(ds_bios,plays_full,plays_train, norm_plays_full,model,artist_index, index_artist)

    save_path='../fake_results/content_based/'
    for k in [5,10,100,200,500]:
            save_object(diversities[k],save_path+'diversity_'+str(k)+'.pkl')
            save_object(precisions[k],save_path+'precision_list_'+str(k)+'.pkl')
            save_object(ndcgs[k],save_path+'ndcg_list_'+str(k)+'.pkl')
            save_object(mrrs[k],save_path+'mrr_list_'+str(k)+'.pkl')

    
dataset_path= '../dataset/'
# dataset_path= '../fake_dataset/'
evaluate(dataset_path)

