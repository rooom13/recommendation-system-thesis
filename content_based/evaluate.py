import pandas as pd


from TfidfRecommender import TfidfRecommender

import sys

fakeDataset = False
dataset_path = '../fake_dataset/' if fakeDataset else '../dataset/'
bios_path = dataset_path + 'bios.txt'

ds = pd.read_csv(bios_path,sep='\t') 

print('GO')
cb_recommender = TfidfRecommender(ds['bio'].tolist())


query_indices =[0,1,2,9]

print('Recomending')
rec_indices = cb_recommender.recommend_similars(query_indices,10)

# indices to ids, 
rec_artists = [ds.iloc[i]['id'] for i in rec_indices]

query_artists = [ds.iloc[i]['id'] for i in query_indices]
print('Because listening to:')
for artist in query_artists:
    print('- '+artist, ds[ds['id'] == artist]['bio'].tolist()[0][:100] )
print('It\'s recommended:')        
for artist in rec_artists:
    print('- '+artist, ds[ds['id'] == artist]['bio'].tolist()[0][:400] )
