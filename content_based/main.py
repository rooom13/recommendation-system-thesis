# from process_data.bios_to_table import bios_to_table
from evaluate import evaluate
# from view_metrics import view_metrics
import sys


fakeDataset = True
dataset_path = '../fake_dataset/' if fakeDataset else '../dataset/'
results_path =  '../fake_results/' if fakeDataset else '../results/'


# bios_to_table(dataset_path)


try:
    kk = [int(sys.argv[1])]
except:
    kk = [5,10,100,200,500]
    kk = [5]
evaluate(dataset_path, results_path,kk=kk)
# view_metrics(results_path,kk=kk, showPrecision=True, showNdcg=True, showMrr=True, showPlots=False)

