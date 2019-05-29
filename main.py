print('Go!',end='...')
# from collaborative_filtering.read_data import read_data as read_triplets
# from collaborative_filtering.scale_data import scale_data
# from collaborative_filtering.scale_data import scale_data
# from collaborative_filtering.generate_model import generate_model

# from content_based.bios_to_table import bios_to_table as read_bios

# from get_dataset import getDataset as get_dataset
# from evaluate import evaluate
# from evaluate2 import evaluate as evaluate2

from view_metrics import view_metrics
fakeDataset = False
datasetPath = './fake_dataset/' if fakeDataset else './dataset/'
resultsPath = './fake_results/' if fakeDataset else './results/'


try:
    kk = [int(sys.argv[1])]
except:
    kk = [5,10,100,200,500]

metrics = {
    'map': True, 
    'diversity': True, 
    'ndcg': True,
    'mrr': True,
    'rnd': False,
    'ub': False
    }
methods = {
    'cf': False,
    'cb': False,
    'hybrid': True
}


# if not fakeDataset: get_dataset()

# read_triplets(datasetPath)
# scale_data(datasetPath)
# read_bios(datasetPath)


# generate_model(datasetPath)

# print('evaluate!')
#evaluate(datasetPath,resultsxXPath,kk=kk, metrics=metrics, methods=methods)
# evaluate2(datasetPath,resultsPath,kk=kk, metrics=metrics, methods=methods)
view_metrics(resultsPath, kk=kk, metrics=metrics,methods=methods)
