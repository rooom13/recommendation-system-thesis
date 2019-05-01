import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import glob

fakeDataset = True
dataset_path = '../fake_dataset/' if fakeDataset else '../dataset/'
bios_path = dataset_path+'bios/'




#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
# tfidf = TfidfVectorizer(stop_words='english')

files= {}
for filename in glob.glob( bios_path + '*.txt'):
    
    with open(filename, "r") as file:
        
        artist = filename[len(bios_path):-len('.txt')]
        files[artist] = file.read()

a = pd.DataFrame.from_dict(files, orient='index')
print(a)

    # print(pd.read_csv(filename, sep="1234321"))


# l = [pd.read_csv(filename) for filename in glob.glob( bios_path + '*.txt')]
# print(l)
# df = pd.concat(l, axis=0)

# print(df)

# 
# txt1 = ['His smile was not perfect', 'His smile was not not not not the perfect', 'she not sang']
# 
# tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word', stop_words='english')
# txt_fitted = tf.fit(txt1)
# txt_transformed = txt_fitted.transform(txt1)
# print (tf.vocabulary_)