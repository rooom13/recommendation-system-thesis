
import matplotlib.pyplot as plt
from ReadSave import * 

import numpy as np

def avg(d):
        return sum(d)/len(d)
        
DEFAULT_METHODS= {
    'cb': True,
    'cf': True,
    'hb': True
}    

DEFAULT_METRICS = {
    'map': True, 
    'diversity': True, 
    'ndcg': True, 
    'mrr': True,
    'rnd': True,
    'ub': True,
    }
def view_metrics(resultsPath,kk=[5,10,100,200,500],metrics=DEFAULT_METRICS, methods=DEFAULT_METHODS, showPlots=True):

    results_cf = resultsPath + 'collaborative_filtering/' 
    results_cb = resultsPath + 'content_based/' 
    results_hybrid = resultsPath + 'hybrid/' 

    show_cb = methods['cb']
    show_cf = methods['cf']
    show_hb = methods['hb']

    show_map = metrics['map']
    show_div = metrics['diversity']
    show_ndcg = metrics['ndcg']
    show_mrr = metrics['mrr']
    show_ub = metrics['ub']
    show_rnd = metrics['rnd']
    
    

    randomBaselines_list=[]
    upper_bound_list=[]
    cf_map_list=[]
    cb_map_list=[]
    hybrid_map_list=[]

    cf_ndcg_list=[]
    cb_ndcg_list=[]
    hybrid_ndcg_list=[]

    cf_mrr_list=[]
    cb_mrr_list=[]
    hybrid_mrr_list=[]

    cf_diversity_list = []
    cb_diversity_list = []
    hybrid_diversity_list = []
    
    k = [5]

    for k in kk:
        if(show_rnd):
            randomBaselines  = read_object(results_cf + 'rnd_baseline_list_'+ str(k) +'.pkl')
            randomBaselines_list.append(avg(randomBaselines))
        if(show_ub):
            upperBounds = read_object(results_cf + 'upper_bound_list_'+ str(k) +'.pkl')
            upper_bound_list.append(avg(upperBounds))
        
        if(show_map): 
            if(show_cf):
                precisions_cf = read_object(results_cf + 'precision_list_'+ str(k) +'.pkl')
                cf_map_list.append(avg(precisions_cf))
            if(show_cb):
                precisions_cb = read_object(results_cb + 'precision_list_'+ str(k) +'.pkl')
                cb_map_list.append(avg(precisions_cb))
            if(show_hb):
                precisions_hybrid = read_object(results_hybrid + 'precision_list_'+ str(k) +'.pkl')
                hybrid_map_list.append(avg(precisions_hybrid))

        
        if(show_ndcg): 
            if(show_cf):
                ndcg_cf = read_object(results_cf + 'ndcg_list_'+ str(k) +'.pkl')
                cf_ndcg_list.append(avg(ndcg_cf))
            if(show_cb):
                ndcg_cb = read_object(results_cb + 'ndcg_list_'+ str(k) +'.pkl')
                cb_ndcg_list.append(avg(ndcg_cb))
            if(show_hb):
                ndcg_hybrid = read_object(results_hybrid + 'ndcg_list_'+ str(k) +'.pkl')
                hybrid_ndcg_list.append(avg(ndcg_hybrid))
        
        if(show_mrr):

            if(show_cf):
                mrr_cf = read_object(results_cf + 'mrr_list_'+ str(k) +'.pkl')
                cf_mrr_list.append(avg(mrr_cf))
            if(show_cb):
                mrr_cb = read_object(results_cb + 'mrr_list_'+ str(k) +'.pkl')
                mrr_cb =np.nan_to_num(mrr_cb)
                cb_mrr_list.append(avg(mrr_cb))
            if(show_hb):
                mrr_hybrid = read_object(results_hybrid + 'mrr_list_'+ str(k) +'.pkl')
                mrr_hybrid =np.nan_to_num(mrr_hybrid)
                hybrid_mrr_list.append(avg(mrr_hybrid))
        
        if(show_div):
            if(show_cf):
                diversity_cf = read_object(results_cf + 'diversity_'+ str(k) +'.pkl')
                cf_diversity_list.append(len(diversity_cf))
            if(show_cb):
                diversity_cb = read_object(results_cb + 'diversity_'+ str(k) +'.pkl')
                cb_diversity_list.append(len(diversity_cb))
            if(show_hb):
                diversity_hybrid = read_object(results_hybrid + 'diversity_'+ str(k) +'.pkl')
                hybrid_diversity_list.append(len(diversity_hybrid))

    
    TAB = '\t\t'
    NL = '\n'
    DEC = 4

    print('Evaluations results for k =',kk,end=':'+NL+NL)


    # Precision
    if(show_cf and show_map):
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
    if(show_cb and show_map):
        print('CB',end=TAB)
        for i in range(0, len(kk) ) :
            score = round(cb_map_list[i],DEC)
            print(score,end=TAB)
        print(NL,end=TAB)
    if(show_hb and show_map):
        print('HB',end=TAB)
        for i in range(0, len(kk) ) :
            score = round(hybrid_map_list[i],DEC)
            print(score,end=TAB)
        print(NL,end=TAB)

    if(show_rnd):
        print('-',end=NL+TAB)
        print('rnd',end=TAB)
        for i in range(0, len(kk) ) :
            score = round(randomBaselines_list[i],DEC)
            print(score,end=TAB)
        print(NL,end=TAB)
    if(show_ub):
        print('ub',end=TAB)
        for i in range(0, len(kk) ) :
            score = round(upper_bound_list[i],DEC)
            print(score,end=TAB)
    print('')

    if(show_ndcg):

        # nDCG
        print('Normalized Discounted Cumulative Gain (nDCG):')
        print('',end=TAB)
        print('Method',end=TAB)         
        for k in kk:
            print('k='+str(k),end=TAB)
        
        print(NL,end=TAB)
        if(show_cf):
            print('CF',end=TAB)
            for i in range(0, len(kk) ) :
                score = round(cf_ndcg_list[i],DEC)
                print(score,end=TAB)
            print(NL,end=TAB)
        if(show_cb):
            print('CB',end=TAB)
            for i in range(0, len(kk) ) :
                score = round(cb_ndcg_list[i],DEC)
                print(score,end=TAB)
            print(NL,end=TAB)
        if(show_hb):
            print('HB',end=TAB)
            for i in range(0, len(kk) ) :
                score = round(hybrid_ndcg_list[i],DEC)
                print(score,end=TAB)
        print('')

    if(show_mrr):

        # MRR
        print('Mean Reciprocal Ranking (MRR):')
        print('',end=TAB)
        print('Method',end=TAB)         
        for k in kk:
            print('k='+str(k),end=TAB)
        
        print(NL,end=TAB)
        if(show_cf):
            print('CF',end=TAB)
            for i in range(0, len(kk) ) :
                score = round(cf_mrr_list[i],DEC)
                print(score,end=TAB)
            print(NL,end=TAB)
        if(show_cb):            
            print('CB',end=TAB)
            for i in range(0, len(kk) ) :
                score = round(cb_mrr_list[i],DEC)
                print(score,end=TAB)
            print(NL,end=TAB)
        if(show_hb):            
            print('HB',end=TAB)
            for i in range(0, len(kk) ) :
                score = round(hybrid_mrr_list[i],DEC)
                print(score,end=TAB)
        print('')
    if(show_div):

        # Diversity
        print('Diversity (# diferent recommended artists ):')
        print('',end=TAB)
        print('Method',end=TAB)         
        for k in kk:
            print('k='+str(k),end=TAB)
        
        print(NL,end=TAB)
        if(show_cf):
            print('CF',end=TAB)
            for i in range(0, len(kk) ) :
                score = cf_diversity_list[i]
                print(score,end=TAB)
            print(NL,end=TAB)
        if(show_cb):
            print('CB',end=TAB)
            for i in range(0, len(kk) ) :
                score = cb_diversity_list[i]
                print(score,end=TAB)
            print(NL,end=TAB)
        if(show_hb):
            print('HB',end=TAB)
            for i in range(0, len(kk) ) :
                score = hybrid_diversity_list[i]
                print(score,end=TAB)
        print('')


    if not showPlots:
        return

    else:
        if show_map:
            if show_cf:
                plt.plot(kk, cf_map_list,label='cf')
            if show_hb:
                plt.plot(kk, hybrid_map_list,label='hb')
            if show_cb:
                plt.plot(kk, cb_map_list,label='cb')
            if show_rnd:
                plt.plot(kk, randomBaselines_list,'--r',label='rnd')
            if show_cb:
                plt.plot(kk, upper_bound_list,'--g',label='cb')
            plt.xlabel('k')
            plt.ylabel('MAP')
            plt.title('MAP vs k')
            plt.legend()
  
            plt.show()

        if show_div:
            if show_cf:
                plt.plot(kk, cf_diversity_list,label='cf')
            if show_hb:
                plt.plot(kk, hybrid_diversity_list,label='hb')
            if show_cb:
                plt.plot(kk, cb_diversity_list,label='cb')
            plt.xlabel('k')
            plt.ylabel('Diversity')
            plt.title('Diversity vs k')
            plt.legend()

            plt.show()

        if show_ndcg:
            if show_cf:
                plt.plot(kk, cf_ndcg_list,label='cf')
            if show_hb:
                plt.plot(kk, hybrid_ndcg_list,label='hb')
            if show_cb:
                plt.plot(kk, cb_ndcg_list,label='cb')
            plt.xlabel('k')
            plt.ylabel('nDCG')
            plt.title('nDCG vs k')
            plt.legend()
            plt.show()

        if show_mrr:
            if show_cf:
                plt.plot(kk, cf_mrr_list,label='cf')
            if show_hb:
                plt.plot(kk, hybrid_mrr_list,label='hb')
            if show_cb:
                plt.plot(kk, cb_mrr_list,label='cb')
            plt.xlabel('k')
            plt.ylabel('MRR')
            plt.title('MRR vs k')
            plt.legend()

            plt.show()



# view_metrics('./results/')