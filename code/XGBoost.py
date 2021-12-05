import torch.utils.data
import xgboost as xgb
import time
import numpy as np
from utils.DataPreprocess import standardize_data, get_pretrained_feature_from_h5, SatelliteSet
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import pickle
from sklearn.model_selection import GridSearchCV


class XGBModel(object):
    def __init__(self):
        # a random number generator for reproducibility
        self.rng = np.random.default_rng(seed=0)

        self.model = xgb.XGBRegressor(max_depth=8,
                                      learning_rate=0.001,
                                      n_estimators=100,
                                      silent=True,
                                      objective='reg:squarederror',
                                      min_child_weight=3
                                      )

    def fit(self, train_x, train_y, val_x, val_y):
        """
        Fit the model on the given training data
        :param train_x: Training featrues as a 2d Numpy array of shape(NUM_SAMPLES,NUM_FEATURES)
        :param train_y: Training canopy height as a 1d Numpy array of shape(NUM_SAMPLESM,)
        """
        train_x = standardize_data(train_x)
        val_x = standardize_data(val_x)

        self.model.fit(train_x, train_y, eval_metric='rmse', verbose=True, eval_set=[(val_x, val_y)],
                       early_stopping_rounds=100)

    def cv_fit(self, train_x, train_y):
        train_x = standardize_data(train_x)
        cv_params = {'max_depth': [4, 6, 8],
                     'min_child_weight': [1, 3, 5, 7]}
        self.model = xgb.XGBRegressor(learning_rate=0.3,
                                      n_estimators=100,
                                      silent=True,
                                      objective='reg:squarederror',
                                      nthread=-1,
                                      gamma=0.1,
                                      max_delta_step=0,
                                      subsample=0.8,
                                      colsample_bytree=0.8,
                                      colsample_bylevel=1,
                                      reg_alpha=1,
                                      reg_lambda=1,
                                      seed=2021,
                                      missing=None)
        gbm = GridSearchCV(self.model, param_grid=cv_params,
                           scoring="neg_mean_squared_error",
                           verbose=True)
        gbm.fit(train_x, train_y)
        print(gbm.cv_results_)
        print(gbm.best_params_)
        print(gbm.best_score_)

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


def fit_once(h5_path, num_feature=16, multiple=False):
    x_data, y_data = get_pretrained_feature_from_h5(h5_path, num_feature=num_feature, multiple=multiple)
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, train_size=0.8)
    print(x_train.shape)
    print(y_train.shape)
    model = XGBModel()
    print("Start fitting the model...")
    start = time.time()
    model.fit(x_train, y_train, x_val, y_val)
    # model.cv_fit(x_train,y_train)
    print(f"Finish fitting in {time.time() - start} seconds.")

    start = time.time()
    y_pred = model.predict(x_val)
    print(f"Finish predicting in {time.time() - start} seconds.")
    mae, rmse = model.evaluation(y_val, y_pred)
    print("Validation: MAE {}, RMSE {}".format(mae, rmse))


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


def train_process(model, x, y, xgb_params):
    x, y = batch_data_preprocess(x, y)
    # Train
    model = xgb.train(params=xgb_params, dtrain=xgb.DMatrix(x, y), xgb_model=model)  # xgboost
    # Train score
    y_pred = model.predict(xgb.DMatrix(x))
    train_score = metrics.mean_squared_error(y, y_pred, squared=False)
    del x, y, y_pred
    return model, train_score


def do_validation(model, dataloader):
    # Validation score
    for x_val, y_val in dataloader:
        x_val, y_val = batch_data_preprocess(x_val, y_val)
        y_val_pred = model.predict(xgb.DMatrix(x_val))
        del x_val
        RMSE = metrics.mean_squared_error(y_true=y_val, y_pred=y_val_pred, squared=False)
        del y_val_pred, y_val

    return RMSE


def partial_fit(file_path, multiple=True, num_features=16):
    CHECKPOINT_PATH = "checkpoints" + str(num_features) + "//"
    BATCH_SIZE = 200
    PATH = file_path
    xgb_params = {
        'objective': "reg:squarederror",
        'learning_rate': 0.001,
        'max_depth': 3,
        'min_child_weight': 1
    }

    dset = SatelliteSet(PATH, multiple=multiple, num_feature=num_features)
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
    reg = None
    best_model = None
    TRAIN_BATCH_STEP = 5
    VAL_BATCH_STEP = 5
    batch_count = 0
    start_train = time.time()
    for x, y in tqdm(train_loader):
        reg, score = train_process(reg, x, y, xgb_params)
        if score < best_score:
            print(score)
            best_score = score
            best_model = reg
            with open(os.path.join(CHECKPOINT_PATH, 'PartialFitXGB_checkpoint.pkl'), 'wb') as f:
                pickle.dump(reg, f)
        batch_count += 1
        if batch_count % TRAIN_BATCH_STEP == 0:
            print(f"Batch {batch_count} train accuracy: {score}")
        if batch_count % VAL_BATCH_STEP == 0:
            rmse = do_validation(reg, validation_loader)
            print(f"Batch {batch_count} validation score: \nrmse - {rmse}")
    print(f"Training time: {time.time() - start_train}")
    with open(os.path.join(CHECKPOINT_PATH, 'PartialFitXGB_last.pkl'), 'wb') as f:
        pickle.dump(reg, f)
    rmse = do_validation(best_model, validation_loader)
    print(f"Final validation score: RMSE  {rmse}")


if __name__ == "__main__":
    h5_path = r"D:\II_LAB2_DATA\c16"
    partial_fit(h5_path, multiple=True, num_features=16)
