import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ReadSave import *
import sys


fakeDataset = True
dataset_path = '../fake_dataset/' if fakeDataset else '../dataset/'
bios_path = dataset_path + 'bios.txt'
ds = pd.read_csv(bios_path,sep='\t') 
tf = TfidfVectorizer(analyzer='word',  min_df=0, stop_words='english') # no ngrams

tfidf_matrix = tf.fit_transform(ds['bio'])

cosine_similarities = cosine_similarity(tfidf_matrix,tfidf_matrix)
def item(id):
    return ds.loc[ds['id'] == id]['bio'].tolist()[0]

def calcuate_similarities(artistid, k):
    # which index is artistid in df 
    artist_index = ds.index[ds['id'].str.match(artistid)].tolist()[0]
    # get k sorted similar artists indexes
    similar_indices = cosine_similarities[artist_index].argsort()[:-(k+1):-1][1:]
    # which artists are those indexes
    similar_artists = [(cosine_similarities[artist_index][i], ds['id'][i]) for i in similar_indices]
    return similar_artists

def recommend(id, k):
    if (k == 0):
        print('Unable to recommend any book as you have not chosen the kber of book to be recommended')
    else:
        print('Showing ' +str(k)+' recommendations similar to ' + item(id)[:20])        
    print('----------------------------------------------------------')
    recs = calcuate_similarities(id,k)
    for rec in recs:
        print('- ' + item(rec[1])[:20] + ' (score:' + str(rec[0]) + ')')

recommend('artist1',6)