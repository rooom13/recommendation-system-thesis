
import matplotlib.pyplot as plt
import pickle
# Storing and readind objects
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# load .pkl object
def read_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def print_histogram(x,k):
        num_bins = 10
        n, bins, patches = plt.hist(x, num_bins)
        plt.xlabel('nDCG score')
        plt.ylabel('user frequency')
        plt.title('Histogram of nDCG for k='+k)
        plt.grid(True)
        plt.yscale('log')
        plt.show()
        

storePath = './precomputed_data/' 


ndcg_k5 = read_object(storePath + 'ndcg_5.pkl')
ndcg_k10 = read_object(storePath + 'ndcg_10.pkl')
ndcg_k15 = read_object(storePath + 'ndcg_15.pkl')

kk = {'5': ndcg_k5, '10': ndcg_k10, '15': ndcg_k15}

for k, data in kk.items():
        print(k)
        print_histogram(data, k)