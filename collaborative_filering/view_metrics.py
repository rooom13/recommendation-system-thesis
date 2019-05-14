
import matplotlib.pyplot as plt
from ReadSave import * 


def print_histogram(x,bins=20, title='Histogram',xlabel='score', ylabel='# users' ):
        n, bins, patches = plt.hist(x, bins)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.yscale('log')
        plt.show()
        
def avg(d):
        return sum(d)/len(d)
        

storePath = './precomputed_data/' 


kk= [5,10,100]

for k in kk:
        precisions = read_object(storePath + 'precision_list_'+ str(k) +'.pkl')
        prec_avg = round(avg(precisions),4)
        print_histogram(precisions,bins=25,title='Histogram Precision@'+str(k)+' avg=' + str(prec_avg),xlabel='score', ylabel='# users')
        
        mrrs = read_object(storePath + 'mrr_list_'+ str(k) +'.pkl')
        mrrs_avg = round(avg(mrrs),4)
        print_histogram(mrrs,bins=25,title='Histogram MRR@'+str(k)+ ' avg='+str(mrrs_avg),xlabel='score', ylabel='# users')
        
        ndgcs = read_object(storePath + 'ndcg_list_'+ str(k) +'.pkl')
        ndcg_avg = round(avg(ndgcs),4)
        print_histogram(ndgcs,bins=25,title='Histogram ndcg@'+str(k)+' avg='+str(ndcg_avg),xlabel='score', ylabel='# users')





