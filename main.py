print('Go!',end='...')
from collaborative_filtering.read_data import split_data
from collaborative_filtering.scale_data import scale_data
from collaborative_filtering.generate_model import generate_model

from content_based.bios_to_table import bios_to_table as read_bios
from content_based.generate_tfIdfRecommender import generate_tfIdfRecommender

from get_dataset import getDataset as get_dataset
from evaluate import evaluate

from data_visualization.view_plays_freq import view_plays_freq
from data_visualization.view_metrics import view_metrics


fakeDataset = True
datasetPath = './fake_dataset/' if fakeDataset else './dataset/'
resultsPath = './fake_results/' if fakeDataset else './results/'

kk = [5 ,10 ,100,200,500]

metrics = {
    'map': True, 
    'diversity': True, 
    'ndcg': True,
    'mrr': True,
    'rnd':  True,
    'ub': True
    }
methods = {
    'cf': True,
    'cb': True,
    'hb': True
}


visualizeDataset = True

if not fakeDataset: get_dataset()

# split_data(datasetPath)

if visualizeDataset:
    view_plays_freq(datasetPath,False)
    view_plays_freq(datasetPath)

# scale_data(datasetPath)
# read_bios(datasetPath)
# generate_model(datasetPath)
# generate_tfIdfRecommender(datasetPath)

print('evaluate!')
evaluate(datasetPath,resultsPath,kk=kk, metrics=metrics, methods=methods)

# view_metrics(resultsPath, kk=kk, metrics=metrics,methods=methods, showPlots=False)
