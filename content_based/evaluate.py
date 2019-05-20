import pandas as pd
from ReadSave import *
import metrics

import numpy as np
from TfidfRecommender import TfidfRecommender

import sys
import os

# Just for a prettier matrix print
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})


def print_progress(text, completed, new_completed):
     if (new_completed > completed): 
            completed = new_completed
            sys.stdout.write('\r'+text+ str(completed) + ' %' )
            sys.stdout.flush()


def loadData(dataset_path):
    print('Evaluation started, loading data...', end='')
    precomputed_path =  dataset_path + 'precomputed_data/' 
    plays_full_path = precomputed_path + 'plays_full.pkl'
    plays_train_path = precomputed_path + 'plays_train.pkl'
    norm_plays_full_path = precomputed_path + 'norm_plays_full.pkl'
    artists_indices_path = precomputed_path + 'artist_index_index_artist.pkl'
    bios_path = dataset_path + 'bios.txt'

 
    plays_full = read_object(plays_full_path)
    print('1/5',end='...')
    plays_train = read_object(plays_train_path)
    print('2/5',end='...')
    norm_plays_full = read_object(norm_plays_full_path)
    print('3/5', end='...')
    ds = pd.read_csv(bios_path,sep='\t') 
    print('4/5', end='...')
    artist_index, index_artist = read_object(artists_indices_path)
    print('5/5')



    return plays_full.tocsr(), plays_train.tocsr(), norm_plays_full.tocsr(),ds, artist_index, index_artist


def get_scores(ds_bios,plays_full,plays_train,norm_plays_full, model,artist_index, index_artist,k=5):
    
    NUSERS,NARTISTS = plays_full.shape    
    
    precision_list = []
    mrr_list = []
    ndcg_list = []

    completed = 0
    new_completed = 0

    for user_id in range(0,NUSERS):
        new_completed = (user_id +1)/ (NUSERS) * 100
        print_progress('\tEvaluating k=' + str(k) + '\t  ', completed ,new_completed  )
    
        # user id mapped to usernames
        user_history = [index_artist[artistid] for artistid in (plays_train[user_id] > 1).nonzero()[1] ]

        # whichs indices in bios
        history_index_bios = ds_bios[ds_bios['id'].isin(user_history)].index.values
        # recommend
        rec_indices = model.recommend_similars(history_index_bios,k)

        # which artists id are those indices 
        rec_artists = [ds_bios.iloc[i]['id'] for i in rec_indices]

        scores = []
        relevants = []


        for artist in rec_artists:
                artist_id = artist_index[artist]
                ground_truth = plays_full[user_id,artist_id]
                # print(ground_truth) 
                relevants.append(1 if ground_truth > 1 else 0) 

                norm_ground_truth = norm_plays_full[user_id,artist_id]
                scores.append(norm_ground_truth)





        precision_list.append(sum(relevants)/k)
        mrr_list.append(metrics.mean_reciprocal_rank(relevants))
        ndcg_list.append(metrics.ndcg_at_k(scores, k))
        """
        
        query_artists = [ds_bios.iloc[i]['id'] for i in history_index_bios]
        print('Because listening to:')
        for artist in query_artists:
                print('- '+artist, ds_bios[ds_bios['id'] == artist]['bio'].tolist()[0][:100] )
        print('It\'s recommended:')        
        for artist in rec_artists:
                print('- '+artist, ds_bios[ds_bios['id'] == artist]['bio'].tolist()[0][:100] )
        """
    return ndcg_list,precision_list,mrr_list

    


def evaluate(dataset_path,results_path, kk=[10,100,200]):

    if not os.path.exists(results_path):
            os.mkdir(results_path)

    results_path = results_path + 'content_based/' 
    
    if not os.path.exists(results_path):
            os.mkdir(results_path)

    plays_full, plays_train, norm_plays_full, ds_bios, artist_index, index_artist = loadData(dataset_path)

    model = TfidfRecommender(ds_bios['bio'].tolist())

  

    for k in kk:
        ndcg_list, precision_list, mrr_list = get_scores(ds_bios,plays_full,plays_train, norm_plays_full,model,artist_index, index_artist, k=k)
        save_object(ndcg_list,results_path+'ndcg_list_'+str(k)+'.pkl')
        save_object(precision_list,results_path+'precision_list_'+str(k)+'.pkl')
        save_object(mrr_list,results_path+'mrr_list_'+str(k)+'.pkl')

    


fakeDataset = True
dataset_path = '../fake_dataset/' if fakeDataset else '../dataset/'
results_path =  '../fake_results/' if fakeDataset else '../results/'

kk = [5] 

evaluate(dataset_path, results_path,kk=kk)
