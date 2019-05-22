
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
        
DEFAULT_METHODS= {
    'cf': True,
    'cb': True
}    

DEFAULT_METRICS = {
    'map': False, 
    'diversity': False, 
    'ndcg': False, 
    'mrr': False,
    'rnd': True,
    'ub': True,
    }
def view_metrics(resultsPath,kk=[10,100,200],metrics=DEFAULT_METRICS, methods=DEFAULT_METHODS, showPrecision=True, showNdcg=True, showMrr=True, showPlots=False):

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

        cf_diversity_list = []
        cb_diversity_list = []
        

        for k in kk:
            if(metrics['rnd']):
                randomBaselines  = read_object(results_cf + 'rnd_baseline_list_'+ str(k) +'.pkl')
                randomBaselines_list.append(avg(randomBaselines))
            
            if(metrics['ub']):
                upperBounds = read_object(results_cf + 'upper_bound_list_'+ str(k) +'.pkl')
                upper_bound_list.append(avg(upperBounds))
            
            if(metrics['map']):            
                precisions_cf = read_object(results_cf + 'precision_list_'+ str(k) +'.pkl')
                precisions_cb = read_object(results_cb + 'precision_list_'+ str(k) +'.pkl')
                cf_map_list.append(avg(precisions_cf))
                cb_map_list.append(avg(precisions_cb))
            
            if(metrics['ndcg']):            
                ndcg_cf = read_object(results_cf + 'ndcg_list_'+ str(k) +'.pkl')
                ndcg_cb = read_object(results_cb + 'ndcg_list_'+ str(k) +'.pkl')
                cf_ndcg_list.append(avg(ndcg_cf))
                cb_ndcg_list.append(avg(ndcg_cb))
            
            if(metrics['mrr']):
                mrr_cf = read_object(results_cf + 'mrr_list_'+ str(k) +'.pkl')
                mrr_cb = read_object(results_cb + 'mrr_list_'+ str(k) +'.pkl')
                cf_mrr_list.append(avg(mrr_cf))
                cb_mrr_list.append(avg(mrr_cb))
            
            if(metrics['diversity']):
                diversity_cf = read_object(results_cf + 'diversity_'+ str(k) +'.pkl')
                diversity_cb = read_object(results_cb + 'diversity_'+ str(k) +'.pkl')
                cf_diversity_list.append(len(diversity_cf))
                cb_diversity_list.append(len(diversity_cb))
                # print(diversity_cb)

      
        TAB = '\t\t'
        NL = '\n'
        DEC = 10

        print('Evaluations results for k =',kk,end=':'+NL+NL)


        # Precision
        if(methods['cf'] and metrics['map']):
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
        if(methods['cb'] and metrics['map']):
            print('CB',end=TAB)
            for i in range(0, len(kk) ) :
                score = round(cb_map_list[i],DEC)
                print(score,end=TAB)
            print(NL,end=TAB)

        if(metrics['rnd']):
            print('-',end=NL+TAB)
            print('rnd',end=TAB)
            for i in range(0, len(kk) ) :
                score = round(randomBaselines_list[i],DEC)
                print(score,end=TAB)
            print(NL,end=TAB)
        if(metrics['ub']):
            print('ub',end=TAB)
            for i in range(0, len(kk) ) :
                score = round(upper_bound_list[i],DEC)
                print(score,end=TAB)
        print('')

        if(metrics['ndcg']):

            # nDCG
            print('Normalized Discounted Cumulative Gain (nDCG):')
            print('',end=TAB)
            print('Method',end=TAB)         
            for k in kk:
                print('k='+str(k),end=TAB)
            
            print(NL,end=TAB)
            if(methods['cf']):
                print('CF',end=TAB)
                for i in range(0, len(kk) ) :
                    score = round(cf_ndcg_list[i],DEC)
                    print(score,end=TAB)
                print(NL,end=TAB)
            if(methods['cb']):
                print('CB',end=TAB)
                for i in range(0, len(kk) ) :
                    score = round(cb_ndcg_list[i],DEC)
                    print(score,end=TAB)
            print('')

        if(metrics['mrr']):

            # MRR
            print('Mean Reciprocal Ranking (MRR):')
            print('',end=TAB)
            print('Method',end=TAB)         
            for k in kk:
                print('k='+str(k),end=TAB)
            
            print(NL,end=TAB)
            if(methods['cf']):
                print('CF',end=TAB)
                for i in range(0, len(kk) ) :
                    score = round(cf_mrr_list[i],DEC)
                    print(score,end=TAB)
                print(NL,end=TAB)
            if(methods['cb']):            
                print('CB',end=TAB)
                for i in range(0, len(kk) ) :
                    score = round(cb_mrr_list[i],DEC)
                    print(score,end=TAB)
            print('')
        if(metrics['diversity']):

            # Diversity
            print('Diversity (# diferent recommended artists ):')
            print('',end=TAB)
            print('Method',end=TAB)         
            for k in kk:
                print('k='+str(k),end=TAB)
            
            print(NL,end=TAB)
            if(methods['cf']):
                print('CF',end=TAB)
                for i in range(0, len(kk) ) :
                    score = cf_diversity_list[i]
                    print(score,end=TAB)
                print(NL,end=TAB)
            if(methods['cb']):
                print('CB',end=TAB)
                for i in range(0, len(kk) ) :
                    score = cb_diversity_list[i]
                    print(score,end=TAB)
            print('')

