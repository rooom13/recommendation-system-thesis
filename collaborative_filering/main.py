# from process_data.read_data import read_data
# from process_data.scale_data import scale_data
# from process_data.generate_model import generate_model

from evaluate import evaluate
from view_metrics import view_metrics
fakeDataset = True
kk = [3]
# read_data(fakeDataset)
# scale_data(fakeDataset)
# generate_model(fakeDataset)

evaluate(fakeDataset,kk=kk)

view_metrics(fakeDataset,kk=kk)

