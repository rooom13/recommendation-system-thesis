import sys
from scipy.sparse import coo_matrix, csr_matrix
from numpy import array
import implicit


# initialize ALS mddel model
model = implicit.als.AlternatingLeastSquares(factors=50)

# train the model on a sparse matrix of item/user/confidence weights
"""
item_users (csr_matrix) – Matrix of confidences for the liked items. This matrix should be a csr_matrix where the 
rows of the matrix are the item, 
the columns are the users that liked that item,
and the value is the confidence that the user liked the item.
show_progress (bool, optional) – Whether to show a progress bar during fitting
"""
# sparse matrix
item_user_raw = array([
    # 0  1  2  3 users
    [9, 0, 0, 0],  # item_0 
    [9, 0, 0, 9],  # item_1
    [9, 1, 0, 9],  # 2
    [0, 0, 1, 0],  # 3
    [0, 0, 2, 0],  # 4
    [3, 0, 0, 0],  # 5
    [0, 0, 4, 0],  # 6
    [9, 0, 0, 2],  # 7
    [0, 2, 0, 0],  # 8
    [0, 0, 0, 1],  # 9
])

# convert to compressed sparse row matrix (csr_matrix)
item_user_data = csr_matrix(item_user_raw)

# fit model
model.fit(item_user_data, show_progress=False)

userId = 0
amount = 3
    
print('Predicting', amount, 'items for user', userId)

# recommend items for a user
user_items = item_user_data.T.tocsr()
recomend = model.recommend(userId, user_items, N=amount)

print(recomend)