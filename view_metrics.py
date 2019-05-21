
import matplotlib.pyplot as plt
from ReadSave import * 
import sys


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
        
def view_metrics(resultsPath,kk=[10,100,200], showPrecision=True, showNdcg=True, showMrr=True, showPlots=False):

        results_cf = resultsPath + 'collaborating_filtering/' 
        results_cb = resultsPath + 'content_based/' 

        
      

        randomBaselines_list=[]
        upper_bound_list=[]
        cf_map_list=[]
        cb_map_list=[]

        cf_ndcg_list=[]
        cb_ndcg_list=[]

        cf_mrr_list=[]
        cb_mrr_list=[]
        
        for k in kk:
            randomBaselines = read_object(results_cf + 'rnd_baseline_list_'+ str(k) +'.pkl')
            upperBounds = read_object(results_cf + 'upper_bound_list_'+ str(k) +'.pkl')
            precisions_cf = read_object(results_cf + 'precision_list_'+ str(k) +'.pkl')
            precisions_cb = read_object(results_cb + 'precision_list_'+ str(k) +'.pkl')
            ndcg_cf = read_object(results_cf + 'ndcg_list_'+ str(k) +'.pkl')
            ndcg_cb = read_object(results_cb + 'ndcg_list_'+ str(k) +'.pkl')
            mrr_cf = read_object(results_cf + 'mrr_list_'+ str(k) +'.pkl')
            mrr_cb = read_object(results_cb + 'mrr_list_'+ str(k) +'.pkl')
            
            randomBaselines_list.append(avg(randomBaselines))
            upper_bound_list.append(avg(upperBounds))
            cf_map_list.append(avg(precisions_cf))
            cb_map_list.append(avg(precisions_cb))
            cf_ndcg_list.append(avg(ndcg_cf))
            cb_ndcg_list.append(avg(ndcg_cb))
            cf_mrr_list.append(avg(mrr_cf))
            cb_mrr_list.append(avg(mrr_cb))


        TAB = '\t\t'
        NL = '\n'
        DEC = 4

        print('Evaluations results for k =',kk,end=':'+NL+NL)

        # Precision
        print('Mean Average Precision: (MAP)')
        print('',end=TAB)
        print('Method',end=TAB)         
        for k in kk:
            print('k='+str(k),end=TAB)
        
        print(NL,end=TAB)
        print('CF',end=TAB)
        for i in range(0, len(kk) ) :
            score = round(cf_map_list[i],DEC)
            print(score,end=TAB)
        print(NL,end=TAB)
        print('CB',end=TAB)
        for i in range(0, len(kk) ) :
            score = round(cb_map_list[i],DEC)
            print(score,end=TAB)
        print(NL,end=TAB)
        print('-',end=NL+TAB)
        print('rnd',end=TAB)
        for i in range(0, len(kk) ) :
            score = round(randomBaselines_list[i],DEC)
            print(score,end=TAB)
        print(NL,end=TAB)
        print('ub',end=TAB)
        for i in range(0, len(kk) ) :
            score = round(upper_bound_list[i],DEC)
            print(score,end=TAB)
        print('')

        # nDCG
        print('Normalized Discounted Cumulative Gain (nDCG):')
        print('',end=TAB)
        print('Method',end=TAB)         
        for k in kk:
            print('k='+str(k),end=TAB)
        
        print(NL,end=TAB)
        print('CF',end=TAB)
        for i in range(0, len(kk) ) :
            score = round(cf_ndcg_list[i],DEC)
            print(score,end=TAB)
        print(NL,end=TAB)
        print('CB',end=TAB)
        for i in range(0, len(kk) ) :
            score = round(cb_ndcg_list[i],DEC)
            print(score,end=TAB)
        print('')



         # MRR
        print('Mean Reciprocal Ranking (MRR):')
        print('',end=TAB)
        print('Method',end=TAB)         
        for k in kk:
            print('k='+str(k),end=TAB)
        
        print(NL,end=TAB)
        print('CF',end=TAB)
        for i in range(0, len(kk) ) :
            score = round(cf_mrr_list[i],DEC)
            print(score,end=TAB)
        print(NL,end=TAB)
        print('CB',end=TAB)
        for i in range(0, len(kk) ) :
            score = round(cb_mrr_list[i],DEC)
            print(score,end=TAB)
        print('')

