import pandas as pd


from TfidfRecommender import TfidfRecommender


fakeDataset = True
dataset_path = '../fake_dataset/' if fakeDataset else '../dataset/'
bios_path = dataset_path + 'bios.txt'

ds = pd.read_csv(bios_path,sep='\t') 

cb_recommender = TfidfRecommender(ds['bio'].tolist())


user_query= 'Electronic'
artists_indices = cb_recommender.recommend(user_query,6)

# indices to ids, 
artists = [ds.iloc[i]['id'] for i in artists_indices]

print('Query:', user_query)        
for artist in artists:
    print('- '+artist, ds[ds['id'] == artist]['bio'].tolist()[0][:20] )
