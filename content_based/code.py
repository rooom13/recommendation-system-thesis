import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ReadSave import *

class TfidfRecommender:
    def __init__(self, dataset):
        self.ds = dataset
        self.tfidf_vectorizer = TfidfVectorizer(analyzer='word',  min_df=0, stop_words='english') # no ngrams
        self.tfidf_matrix = tf.fit_transform(ds['bio'])

    def calcuate_similarities(self, tfidf_query, k):
    
        # get K cosine similar artists to query
        retrieved_indices = cosine_similarity(tfidf_query, tfidf_matrix).argsort()[0][:-(k+1):-1]
        # which artists are those indexes
        artists = [ds.iloc[i]['id'] for i in retrieved_indices]
        return artists

    def recommend(self, user_query, k):
        tfidf_query = tfidf_query = tf.transform([user_query])
        rec_artists = self.calcuate_similarities(tfidf_query,k)
        return  rec_artists

def bio(id):
    return ds[ds['id'] == id]['bio'].tolist()[0]




fakeDataset = True
dataset_path = '../fake_dataset/' if fakeDataset else '../dataset/'
bios_path = dataset_path + 'bios.txt'

ds = pd.read_csv(bios_path,sep='\t') 
tf = TfidfVectorizer(analyzer='word',  min_df=0, stop_words='english') # no ngrams

tfidf_matrix = tf.fit_transform(ds['bio'])


user_query= 'Electronic'

artists = recommend(user_query,6)
print(artists)

# print('Query:', user_query)        
# for artist in artists:
    # print('- ' + bio(artist)[:20] )
