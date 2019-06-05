
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

        results_path = resultsPath + 'collaborative_filtering/' 

        print('Evaluations results for k=',kk)
        

        for k in kk:
                
                if (showPrecision):
                
                        precisions = read_object(results_path + 'precision_list_'+ str(k) +'.pkl')
                        randomBaselines = read_object(results_path + 'rnd_baseline_list_'+ str(k) +'.pkl')
                        upperBounds = read_object(results_path + 'upper_bound_list_'+ str(k) +'.pkl')
                        
                        prec_avg = round(avg(precisions),4)
                        rnd_baseline = avg(randomBaselines)
                        upper_bound = round(avg(upperBounds),4)
                        print('\tk='+str(k))
                        print('\t\t- precision:\t'+str(prec_avg))
                        print('\t\trnd_baseline:\t'+str(rnd_baseline))
                        print('\t\tupper_bound:\t'+str(upper_bound))

                        if showPlots:
                                print_histogram(precisions,bins=25,title='Histogram Precision@' + ' μ=' + str(prec_avg),xlabel='score', ylabel='# users')
                
                if (showMrr):
                        mrrs = read_object(results_path + 'mrr_list_'+ str(k) +'.pkl')
                        mrrs_avg = round(avg(mrrs),4)

                        print('\t\t- MRR:\t'+str(mrrs_avg))
                        
                        if showPlots:
                                print_histogram(mrrs,bins=25,title='Histogram MRR@'+str(k)+ ' avg='+str(mrrs_avg),xlabel='score', ylabel='# users')
                
                if (showNdcg):
                        ndgcs = read_object(results_path + 'ndcg_list_'+ str(k) +'.pkl')
                        ndcg_avg = round(avg(ndgcs),4)
                        print('\t\t- nDCG:\t'+str(ndcg_avg))
                        if showPlots:                
                                print_histogram(ndgcs,bins=25,title='Histogram ndcg@'+str(k)+' avg='+str(ndcg_avg),xlabel='score', ylabel='# users')

