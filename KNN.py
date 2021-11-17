from sklearn.neighbors import KNeighborsRegressor
import time
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split

from utils.DataPreprocess import standardize_data
from utils.dataset_windows import SatelliteSet


class KNNModel(object):
    def __init__(self):
        # a random number generator for reproducibility
        self.rng = np.random.default_rng(seed=0)

        self.model = KNeighborsRegressor(n_neighbors=5, n_jobs=-1, algorithm="ball_tree")

    def cv_fit(self, train_x, train_y):
        """
        Fit the model on the given training data using GridSearch
        :param train_x: Training featrues as a 2d Numpy array of shape(NUM_SAMPLES,NUM_FEATURES)
        :param train_y: Training canopy height as a 1d Numpy array of shape(NUM_SAMPLESM,)
        """
        train_x = standardize_data(train_x)

        param_grid = {"n_neighbors": [5, 10, 15, 20, 30]}
        grid_obj = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=3,
                                scoring=["neg_mean_absolute_error"],
                                n_jobs=1,
                                verbose=2,
                                refit="neg_mean_absolute_error")
        grid_obj.fit(train_x, train_y)

        results = pd.DataFrame(pd.DataFrame(grid_obj.cv_results_))
        print("best_index", grid_obj.best_index_)
        print("best_score", grid_obj.best_score_)
        print("best_params", grid_obj.best_params_)

        return grid_obj.best_params_

    def fit(self, train_x, train_y):
        train_x = standardize_data(train_x)
        self.model.fit(train_x, train_y)

    def predict(self, x):
        """
        Predict the canopy height for a given set of images
        :param x:
        :return:
        """

        x = standardize_data(x)
        predictions = self.model.predict(x)
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


def read_train(file_path=r"D:\II_LAB2_DATA\Raw_Feature.npy"):
    with open(file_path, "rb") as f:
        x_train = np.load(f)
        y_train = np.load(f)
    return x_train, y_train


def main():
    # load data
    t1 = time.time()
    h5 = h5py.File(r"./data/data_train_features_c16.hdf5", 'r')
    fea = h5["Features"]
    gt = h5['GT']
    num_feat = fea.shape[1]
    fea = np.transpose(fea, (0, 2, 3, 1))
    x_sample = fea.reshape(-1, 1, num_feat)
    y_flat = np.asarray(gt).reshape(-1, 1)
    x_data = x_sample[np.where(y_flat != -1)][:100000]
    y_data = y_flat[y_flat != -1].reshape(-1, 1)[:100000]

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, train_size=0.8)
    print(x_train.shape)
    print(x_val.shape)
    print(y_train.shape)
    print(y_val.shape)

    # load model
    model = KNNModel()
    print("Start fitting the model...")
    start = time.time()
    model.fit(x_train, y_train)
    print("Finish fitting in {} seconds.".format(time.time() - start))

    start = time.time()
    y_pred = model.predict(x_val)
    print("Finish predicting in {} seconds.".format(time.time() - start))
    mae, rmse = model.evaluation(y_val, y_pred)
    print("Validation: MAE {}, RMSE {}".format(mae, rmse))

    # neigh = KNeighborsRegressor(n_neighbors=5)
    # start = time.time()
    # neigh.fit(x_train, y_train)
    # print(f"Finish fitting in {time.time() - start} seconds.")
    # y_pred = neigh.predict(x_val)
    # mae = metrics.mean_absolute_error(y_true=y_val,y_pred=y_pred)
    # RMSE = metrics.mean_squared_error(y_true=y_val, y_pred=y_pred, squared=False)
    # print(mae)
    # print(RMSE)
    # print("Start fitting the model...")
    # start = time.time()
    # print(f"Finish fitting in {time.time() - start} seconds.")


if __name__ == "__main__":
    main()
