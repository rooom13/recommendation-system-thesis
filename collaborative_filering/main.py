from process_data.read_data import read_data
from process_data.scale_data import scale_data
from process_data.generate_model import generate_model

from ndcg import evaluate_ndcg
from precision import evaluate_precision
fakeDataset = False

# read_data(fakeDataset)
# scale_data(fakeDataset)
# generate_model(fakeDataset)

# evaluate_ndcg(fakeDataset)
evaluate_precision(fakeDataset)