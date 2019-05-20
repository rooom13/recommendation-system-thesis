# from process_data.bios_to_table import bios_to_table
from evaluate import evaluate


fakeDataset = False
dataset_path = '../fake_dataset/' if fakeDataset else '../dataset/'
results_path =  '../fake_results/' if fakeDataset else '../results/'


# bios_to_table(dataset_path)
kk = [5] 
evaluate(dataset_path, results_path,kk=kk)


