import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ReadSave import *
import sys


fakeDataset = False
dataset_path = '../fake_dataset/' if fakeDataset else '../dataset/'
bios_path = dataset_path + 'bios.txt'
ds = pd.read_csv(bios_path,sep='\t') 
tf = TfidfVectorizer(analyzer='word',  min_df=0, stop_words='english') # no ngrams

tfidf_matrix = tf.fit_transform(ds['bio'])

cosine_similarities = cosine_similarity(tfidf_matrix,tfidf_matrix)
# print(cosine_similarities)

results = {} 

for idx, row in ds.iterrows(): 
    # the below code 'similar_indice' stores similar ids based on cosine similarity. sorts them in ascending order. [:-5:-1] is then used so that the indices with most similarity are got. 0 means no similarity and 1 means perfect similarity
    similar_indices = cosine_similarities[idx].argsort()[:-5:-1] #stores 5 most similar books, you can change it as per your needs
    
   
    similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]
    results[row['id']] = similar_items[1:]

save_object(results,'results.pkl')
#below code 'function item(id)' returns a row matching the id along with Book Title. Initially it is a dataframe, then we convert it to a list
def item(id):
    return ds.loc[ds['id'] == id]['bio'].tolist()[0]
def recommend(id, num):
    if (num == 0):
        print("Unable to recommend any book as you have not chosen the number of book to be recommended")
    elif (num==1):
        print("Recommending " + str(num) + " book similar to " + item(id)[:20])
        
    else :
        print("Recommending " + str(num) + " books similar to " + item(id)[:20])
        
    print("----------------------------------------------------------")
    recs = results[id][:num]
    for rec in recs:
        print("- " + item(rec[1])[:20] + " (score:" + str(rec[0]) + ")")

#the first argument in the below function to be passed is the id of the book, second argument is the number of books you want to be recommended

recommend('artist2',3)