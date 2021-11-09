from sklearn.neighbors import KNeighborsRegressor
import numpy as np

X = np.random.randint(0, 100, size = [128*128, 64])
y = np.random.randint(0, 10, size=[128*128])

model = KNeighborsRegressor(n_neighbors=2)

model.fit(X, y)