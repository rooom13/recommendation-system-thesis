import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def read_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

filename = 'raw_object.pkl'

users = ['u1','u2','u3']
artists = ['u1','u2','u3']

sample = (users, artists)


us, ar, pl = read_object(filename)

print(us)
print(ar)
print(pl)