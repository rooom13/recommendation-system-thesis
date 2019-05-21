from collaborative_filtering.read_data import read_data as read_triplets
from collaborative_filtering.scale_data import scale_data
from collaborative_filtering.generate_model import generate_model

from content_based.bios_to_table import bios_to_table as read_bios

from evaluate import evaluate

from view_metrics import view_metrics
fakeDataset = False
datasetPath = './fake_dataset/' if fakeDataset else './dataset/'
resultsPath = './fake_results/' if fakeDataset else './results/'


try:
    kk = [int(sys.argv[1])]
except:
    kk = [5,10,100,200,500]
    kk = [10]

metrics = {
    'map': False, 
    'diversity': False, 
    'ndcg': False, 
    'mrr': False,
    'rnd': True,
    'ub': True
    }
methods = {
    'cf': True,
    'cb': True
}


# read_triplets(datasetPath)
# scale_data(datasetPath)
# read_bios(datasetPath)


# generate_model(datasetPath)
evaluate(datasetPath,resultsPath,kk=kk, metrics=metrics, methods=methods)
view_metrics(resultsPath, kk=kk, metrics=metrics,methods=methods)