import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ReadSave import *
import sys

class TfidfRecommender:
    def __init__(self, dataset):

        self.dataset = dataset
        self.tfidf_vectorizer = TfidfVectorizer(analyzer='word',  min_df=0, stop_words='english') # no ngrams
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(dataset)

    def calcuate_similarities(self, tfidf_query, k, exclude=[]):
        # get K cosine similar indices to query
        artists =  cosine_similarity(tfidf_query, self.tfidf_matrix).argsort()[0][:-(k+1+len(exclude)):-1].tolist()
        
        for i in exclude: 
            try:
                artists.remove(i)
            except:
                pass
   
        return artists
    def recommend_similars(self, artists, k):
        # get tfidf from indexes
        tfidf_artists = [self.tfidf_matrix[index] for index in artists]
        if len(tfidf_artists) == 0:
            return []
        
        #avg tfidf artists
        tfidf_query = sum(tfidf_artists)/len(tfidf_artists)

        rec_artists = self.calcuate_similarities(tfidf_query,k, exclude =artists)
       
        return rec_artists
        
  
    def recommend(self, user_query, k):
        tfidf_query  = self.tfidf_vectorizer.transform([user_query])
        rec_artists = self.calcuate_similarities(tfidf_query,k)
        return  rec_artists

