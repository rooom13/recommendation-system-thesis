
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

def print_histogram(x, title='Histogram',xlabel='score', ylabel='# users' ):
        num_bins = 10
        n, bins, patches = plt.hist(x, num_bins)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.yscale('log')
        plt.show()
        

storePath = './precomputed_data/' 


# ndcg_k10 = read_object(storePath + 'ndcg_10.pkl')
# ndcg_k100 = read_object(storePath + 'ndcg_100.pkl')
# ndcg_k500 = read_object(storePath + 'ndcg_500.pkl')

# precision_k5 = read_object(storePath + 'precision_5.pkl')
precision_k10 = read_object(storePath + 'precision_10.pkl')
# precision_k100 = read_object(storePath + 'precision_100.pkl')
# precision_k500 = read_object(storePath + 'precision_500.pkl')

# ndcg_kk = {'10': ndcg_k10, '100': ndcg_k100, '500': ndcg_k500}
# precision_kk = {'10': precision_k10, '100': precision_k100, '500': precision_k500}

# for k, data in ndcg_kk.items():
#         # print(k)
#         print_histogram(data,title='Histogram of nDCG k=' +  k)

print_histogram(precision_k10,title='Histogram of precision k=10')
# for k, data in precision_kk.items():
        # print(k)
        # print_histogram(data,title='Histogram of precision k=' +  k)