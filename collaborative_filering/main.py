from process_data.read_data import read_data
from process_data.scale_data import scale_data
from process_data.generate_model import generate_model

# from evaluate import evaluate
# from view_metrics import view_metrics

fakeDataset = True
datasetPath = '../fake_dataset/' if fakeDataset else '../dataset/'
resultsPath = '../fake_results/' if fakeDataset else '../results/'


kk = [5,10]



read_data(datasetPath)
scale_data(datasetPath)
generate_model(datasetPath)

# evaluate(datasetPath,resultsPath,kk=kk)

# view_metrics(resultsPath,kk=kk)

