import time
import numpy as np
from sklearn.gaussian_process.kernels import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import metrics

from DataPreprocess import standardize_data


class GPModel(object):
    def __init__(self):
        # a random number generator for reproducibility
        self.rng = np.random.default_rng(seed=0)

        kernel = RBF()
        self.model = GaussianProcessRegressor(kernel=kernel)

    def fit_model(self, train_x, train_y):
        """
        Fit the model on the given training data
        :param train_x: Training featrues as a 2d Numpy array of shape(NUM_SAMPLES,NUM_FEATURES)
        :param train_y: Training canopy height as a 1d Numpy array of shape(NUM_SAMPLESM,)
        """
        train_x = standardize_data(train_x)
        print("Start fitting the model...")
        start = time.time()
        self.model.fit(train_x, train_y)
        print(f"Finish fitting in {time.time() - start} seconds.")

    def predict(self, x):
        """
        Predict the canopy height for a given set of images
        :param x:
        :return:
        """
        gp_mean = np.zeros(x.shape[0], dtype=float)

        gp_mean = self.model.predict(x)
        predictions = gp_mean
        return predictions

    def evaluation(self, y_gt, y_pred):
        """
        Evaluate on the unseen test region, by measuring the mean absolute error (MAE) and the root mean square error (RMSE).
        :param y_pred:
        :param y_gt:
        :return:
        """
        MAE = metrics.mean_absolute_error(y_true=y_gt, y_pred=y_pred)
        RMSE = metrics.mean_squared_error(y_true=y_gt, y_pred=y_pred, squared=False)
        return MAE, RMSE
