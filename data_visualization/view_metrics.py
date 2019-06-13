
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
def view_metrics(results_path,kk=[5,10,100,200,500],metrics=DEFAULT_METRICS, methods=DEFAULT_METHODS, showPlots=True):

    results_rnd = results_path + 'rnd_baseline/' 
    results_ub = results_path + 'upper_bound/' 
    results_cf = results_path + 'collaborative_filtering/' 
    results_cb = results_path + 'content_based/' 
    results_hybrid = results_path + 'hybrid/' 

    methodsKeys = []
    result_paths = []
    if methods['cb']:
            methodsKeys.append('cb')
            result_paths.append(results_path + 'content_based/')
    if methods['cf']:
            methodsKeys.append('cf')
            result_paths.append(results_path + 'collaborative_filtering/')
    if methods['hb']:
            methodsKeys.append('hb')
            result_paths.append(results_path + 'hybrid/')

    show_cb = methods['cb']
    show_cf = methods['cf']
    show_hb = methods['hb']

    show_map = metrics['map']
    show_div = metrics['diversity']
    show_ndcg = metrics['ndcg']
    show_mrr = metrics['mrr']
    show_ub = metrics['ub']
    show_rnd = metrics['rnd']
    
    

    # Global variables for easy backup 
    maps = {}
    mrrs = {}
    ndcgs = {}
    divs = {} 

    randomBaseline_list=[]
    upper_bound_list=[]


    
    for k in kk:
        if(show_rnd):
            randomBaselines  = read_object(results_rnd + 'precision_list_'+ str(k) +'.pkl')
            randomBaseline_list.append(avg(randomBaselines))
        if(show_ub):
            upperBounds = read_object(results_ub + 'precision_list_'+ str(k) +'.pkl')
            upper_bound_list.append(avg(upperBounds))
        

    for method, result_path in zip( methodsKeys,result_paths):
        maps[method]= []
        ndcgs[method]= []
        mrrs[method]= []
        divs[method]= []
        for k in kk:
            if(show_map): 
                    map_list = read_object(result_path + 'precision_list_'+ str(k) +'.pkl')
                    maps[method].append(avg(map_list))
            if(show_ndcg): 
                    ndcg_list = read_object(result_path + 'ndcg_list_'+ str(k) +'.pkl')
                    ndcgs[method].append(avg(ndcg_list))
            if(show_mrr):
                    mrr_list = read_object(result_path + 'mrr_list_'+ str(k) +'.pkl')
                    mrrs[method].append(avg(mrr_list))
            if(show_div):
                    div_list = read_object(result_path + 'diversity_'+ str(k) +'.pkl')
                    divs[method].append(len(div_list))
    
    TAB = '\t\t'
    NL = '\n'
    DEC = 4

    print('Evaluations results for k =',kk,end=':'+NL+NL)

    # MAP
    if show_map:
        print('MAP', end=TAB)
        for k in kk:
            print('k='+str(k),end=TAB)
        print('')
        for method in methodsKeys:
            print(method,end=TAB)
            for i in range(0,len(kk)):
                score = round(maps[method][i],DEC)
                print(score,end=TAB)
            print('')
        print('-')

        # MAP rnd & ub

        methodKeys2,maps2 = [],[]
        if show_rnd:
            methodKeys2.append('rnd')
            maps2.append(randomBaseline_list)
        if show_ub:
            methodKeys2.append('ub')
            maps2.append(upper_bound_list)
        for method, maps2 in zip(methodKeys2, maps2):
            print(method,end=TAB) 
            for i in range(0,len(kk)):
                score = round(maps2[i],DEC)
                print(score,end=TAB)
            print('')
        print('')

    # DIVERSITY
    if show_div:
        print('DIVERS.', end=TAB)
        for k in kk:
            print('k='+str(k),end=TAB)
        print('')
        for method in methodsKeys:
            print(method,end=TAB)
            for i in range(0,len(kk)):
                score = round(divs[method][i],DEC)
                print(score,end=TAB)
            print('')
        print('')
    
    # ndcg
    if show_ndcg:
        print('MRR', end=TAB)
        for k in kk:
            print('k='+str(k),end=TAB)
        print('')
        for method in methodsKeys:
            print(method,end=TAB)
            for i in range(0,len(kk)):
                score = round(ndcgs[method][i],DEC)
                print(score,end=TAB)
            print('')
        print('')
        
    # MRR
    if show_mrr:
        print('MRR', end=TAB)
        for k in kk:
            print('k='+str(k),end=TAB)
        print('')
        for method in methodsKeys:
            print(method,end=TAB)
            for i in range(0,len(kk)):
                score = round(mrrs[method][i],DEC)
                print(score,end=TAB)
            print('')
        print('')

   
    if not showPlots:
        return

    
    if show_map:
        for method, score in zip(methodsKeys, maps):
            plt.plot(kk, maps[method],label=method)
            if show_rnd:
                plt.plot(kk, randomBaseline_list,'--r',label='rnd')
            if show_cb:
                plt.plot(kk, upper_bound_list,'--g',label='cb')
            plt.xlabel('k')
            plt.ylabel('MAP')
            plt.title('MAP vs k')
            plt.legend()
        plt.show()
    
    if show_div:
        for method, score in zip(methodsKeys, divs):
            plt.plot(kk, divs[method],label=method)
            plt.xlabel('k')
            plt.ylabel('Diversity')
            plt.title('Diversity vs k')
            plt.legend()
        plt.show()
    
    if show_ndcg:
        for method, score in zip(methodsKeys, ndcgs):
            plt.plot(kk, ndcgs[method],label=method)
            plt.xlabel('k')
            plt.ylabel('NDCG')
            plt.title('NDCG vs k')
            plt.legend()
        plt.show()
    
    if show_mrr:
        for method, score in zip(methodsKeys, mrrs):
            plt.plot(kk, mrrs[method],label=method)
            plt.xlabel('k')
            plt.ylabel('MRR')
            plt.title('MRR vs k')
            plt.legend()
