import fjaccard
import numpy as np
import pandas as pd

data = pd.DataFrame({
    'user': [1, 1, 1, 1, 2, 2, 2, 0],
    'item': [1, 2, 3, 4, 3, 4, 5, 3]
})
distance_query = fjaccard.FJaccard(data.user.values, data.item.values)
print(distance_query.query(1, 2))
print(distance_query.query(1, 3))
print(distance_query.query_square(np.array([1, 2, 3])))
print(distance_query.query_pairs(np.array([0, 1, 3]), np.array([0, 1, 2, 3, 4, 5])))
