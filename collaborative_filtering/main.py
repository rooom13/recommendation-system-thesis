# from process_data.read_data import read_data
# from process_data.scale_data import scale_data
# from process_data.generate_model import generate_model

import sys
from evaluate import evaluate
# from view_metrics import view_metrics

fakeDataset = False
datasetPath = '../fake_dataset/' if fakeDataset else '../dataset/'
resultsPath = '../fake_results/' if fakeDataset else '../results/'



kk = [10,100,200,500]

kk = [int(sys.argv[1])]



# read_data(datasetPath)
# scale_data(datasetPath)
# generate_model(datasetPath)

evaluate(datasetPath,resultsPath,kk=kk)

# view_metrics(resultsPath,kk=kk)

