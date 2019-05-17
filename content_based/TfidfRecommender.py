from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ReadSave import *

class TfidfRecommender:
    def __init__(self, dataset):

        self.dataset = dataset
        self.tfidf_vectorizer = TfidfVectorizer(analyzer='word',  min_df=0, stop_words='english') # no ngrams
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(dataset)

    def calcuate_similarities(self, tfidf_query, k):
    
        # get K cosine similar indices to query
        return  cosine_similarity(tfidf_query, self.tfidf_matrix).argsort()[0][:-(k+1):-1]

    def recommend(self, user_query, k):
        tfidf_query  = self.tfidf_vectorizer.transform([user_query])
        rec_artists = self.calcuate_similarities(tfidf_query,k)
        return  rec_artists

