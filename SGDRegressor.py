import numpy as np
from sklearn.linear_model import SGDRegressor
import torch.utils.data
import time
from utils.DataPreprocess import standardize_data, get_pretrained_feature_from_h5, SatelliteSet, get_raw_feature_from_npfile
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import pickle
import math
from sklearn.model_selection import GridSearchCV


class SGDRegressorModel(object):
    def __init__(self):
        # a random number generator for reproducibility
        self.rng = np.random.default_rng(seed=0)

        sgd_params = {
            "loss": "squared_loss",
            "penalty": "l2",
            "shuffle": True,
            "warm_start": True,
        }
        self.model = SGDRegressor()
        self.model.set_params(**sgd_params)

    def fit(self, train_x, train_y):
        """
        Fit the model on the given training data
        :param train_x: Training featrues as a 2d Numpy array of shape(NUM_SAMPLES,NUM_FEATURES)
        :param train_y: Training canopy height as a 1d Numpy array of shape(NUM_SAMPLESM,)
        """
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


def batch_data_preprocess(x, y):
    x = np.transpose(x, (0, 2, 3, 1))
    x_flat = x.reshape(-1, 1, x.shape[-1])
    y_flat = y.reshape(-1, 1)
    del x, y
    x_data = x_flat[np.where(y_flat != -1)]
    y_data = y_flat[y_flat != -1].reshape(-1, 1)
    del x_flat, y_flat

    # Standardization (to make data zero mean and unit variance)
    x_scaled = standardize_data(x_data)
    del x_data
    return x_scaled, y_data.numpy()


def train_process(model, x, y):
    x, y = batch_data_preprocess(x, y)
    # Train
    model.partial_fit(x, y)
    # Train score
    y_pred = model.predict(x)
    train_score = metrics.mean_squared_error(y, y_pred, squared=False)
    del x, y, y_pred
    return model, train_score


def do_validation(model, dataloader):
    # Validation score
    MSE = 0
    count = 0
    for x_val, y_val in dataloader:
        count += 1
        x_val, y_val = batch_data_preprocess(x_val, y_val)
        y_val_pred = model.predict(x_val)
        del x_val
        MSE += metrics.mean_squared_error(y_true=y_val, y_pred=y_val_pred)
        del y_val_pred, y_val
    RMSE = math.sqrt(MSE / count)
    return RMSE


def fit_once(h5_path, num_feature=16, multiple=True):
    CHECKPOINT_PATH = "checkpoints16_once"
    x_data, y_data = get_pretrained_feature_from_h5(h5_path, num_feature=num_feature, multiple=multiple)
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, train_size=0.8)

    model = SGDRegressorModel()
    print("Start fitting the model...")
    start = time.time()
    model.fit(x_train, y_train)
    print(f"Finish fitting in {time.time() - start} seconds.")

    start = time.time()
    y_pred = model.predict(x_val)
    print(f"Finish predicting in {time.time() - start} seconds.")

    mae, rmse = model.evaluation(y_val, y_pred)
    print("Validation: MAE {}, RMSE {}".format(mae, rmse))
    with open(os.path.join(CHECKPOINT_PATH, 'SGD.pkl'), 'wb') as f:
        pickle.dump(model, f)


def partial_fit(file_path, num_feature=16, multiple=False):
    CHECKPOINT_PATH = "checkpoints" + str(num_feature) + "//"
    BATCH_SIZE = 200
    PATH = file_path

    dset = SatelliteSet(PATH, num_feature=num_feature, multiple=multiple)
    print(f"Whole data contains {len(dset)} windows.")
    train_dset = torch.utils.data.Subset(dset, range(round(len(dset) * 0.8)))
    validation_dset = torch.utils.data.Subset(dset, range(round(len(dset) * 0.8), len(dset)))
    print(f"Traning data contains {len(train_dset)} windows.")
    print(f"Traning data contains {len(validation_dset)} windows.")
    train_loader = torch.utils.data.DataLoader(train_dset,
                                               batch_size=BATCH_SIZE,
                                               num_workers=0,
                                               shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dset,
                                                    batch_size=len(validation_dset),
                                                    num_workers=0,
                                                    shuffle=True)

    best_score = 999
    best_model = None
    TRAIN_BATCH_STEP = 5
    VAL_BATCH_STEP = 5
    start_train = time.time()
    sgd_params = {
        "loss": "huber",
        "penalty": "l2",
        "shuffle": True,
        "warm_start": True,
    }
    reg = SGDRegressor()
    reg.set_params(**sgd_params)
    batch_count = 0
    print("Start fitting the model...")
    for x, y in tqdm(train_loader):
        reg, score = train_process(reg, x, y)
        if score < best_score:
            print(score)
            best_score = score
            best_model = reg
            with open(os.path.join(CHECKPOINT_PATH, 'SGD_checkpoint.pkl'), 'wb') as f:
                pickle.dump(reg, f)
        batch_count += 1
        if batch_count % TRAIN_BATCH_STEP == 0:
            print(f"Batch {batch_count} train accuracy: {score}")
        if batch_count % VAL_BATCH_STEP == 0:
            rmse = do_validation(reg, validation_loader)
            print(f"Batch {batch_count} validation score: \nrmse - {rmse}")
    print(f"Training time: {time.time() - start_train}")
    with open(os.path.join(CHECKPOINT_PATH, 'PartialFitSGD_last.pkl'), 'wb') as f:
        pickle.dump(reg, f)
    rmse = do_validation(reg, validation_loader)
    print(f"Final validation score: RMSE  {rmse}")


def fit_raw_once(h5_path, num_feature=4, multiple=True):
    CHECKPOINT_PATH = "checkpoints16_once"
    x_data, y_data = get_raw_feature_from_npfile(h5_path)
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, train_size=0.8)

    model = SGDRegressorModel()
    print("Start fitting the model...")
    start = time.time()
    model.fit(x_train, y_train)
    print(f"Finish fitting in {time.time() - start} seconds.")
    with open('SGD_4.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    h5_path = r"D:\jingyli\ImageInterpretation_Regression\data\data_c16_p1_update.hdf5"
    partial_fit(h5_path, multiple=False, num_feature=4)
