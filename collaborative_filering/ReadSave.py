import pickle

# Storing and readind objects
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# load .pkl object
def read_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
