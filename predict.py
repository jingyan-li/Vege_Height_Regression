import math

import h5py
import torch.utils.data
import xgboost as xgb
import time
import numpy as np
from utils.ShallowRegressDataset import standardize_data, get_pretrained_feature_from_h5, SatelliteSet
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import pickle


def batch_data_preprocess(x, y):
    '''
    Flat x and y; Standardize x to [0,1]
    :param x:
    :param y:
    :return:
    '''
    x = np.transpose(x, (0, 2, 3, 1))
    x_flat = x.reshape(-1, x.shape[-1])
    y_flat = y.reshape(-1, 1)
    del x, y
    # x_data = x_flat[np.where(y_flat != -1)]
    # y_data = y_flat[y_flat != -1].reshape(-1, 1)
    # del x_flat, y_flat

    # Standardization (to make data zero mean and unit variance)
    x_scaled = standardize_data(x_flat)
    print(x_scaled.shape)
    print(y_flat.shape)
    del x_flat
    return x_scaled, y_flat.numpy()


def train_process(model, x, y, xgb_params):
    x, y = batch_data_preprocess(x, y)
    # Train
    model = xgb.train(params=xgb_params, dtrain=xgb.DMatrix(x, y), xgb_model=model)  # xgboost
    # Train score
    y_pred = model.predict(xgb.DMatrix(x))
    train_score = metrics.mean_squared_error(y, y_pred, squared=False)
    del x, y, y_pred
    return model, train_score


def do_validation_xgb(model, dataloader, window_size):
    # Validation score
    MSE = 0
    count = 0
    y_predict = np.empty((0, window_size, window_size))
    for x_val, y_val in dataloader:
        count += 1
        x_scaled, y_flat = batch_data_preprocess(x_val, y_val)
        y_val_pred = model.predict(xgb.DMatrix(x_scaled))
        y_val_pred = np.reshape(y_val_pred, (-1, window_size, window_size))
        y_val_pred = np.where(y_val==-1,-1,y_val_pred)
        del x_val
        y_score_gt = y_flat[y_flat != -1]
        y_val_pred_flat = y_val_pred.reshape(-1, 1)
        y_score_pred = y_val_pred_flat[y_flat != -1]
        if len(y_score_gt) > 0:
            MSE += metrics.mean_squared_error(y_true=y_score_gt, y_pred=y_score_pred, squared=True)
        y_predict = np.append(y_predict, y_val_pred, axis=0)
        if count == 1:
            np.savez(f"batch{count}_test.npz", pred=y_predict, gt=y_val)
            # with open("first_batch_test.npy", "wb") as f:
            #     np.save(f, y_val)
            #     np.save(f, y_predict)
            print("save first batch test")

        del y_val_pred, y_val
    RMSE = math.sqrt(MSE/count)
    return RMSE, y_predict


def do_validation(model, dataloader,window_size):
    # Validation score
    MSE = 0
    count = 0
    y_predict = np.empty((0, window_size, window_size))
    for x_val, y_val in dataloader:
        count += 1
        x_scaled, y_flat = batch_data_preprocess(x_val, y_val)
        y_val_pred = model.predict(x_scaled)
        y_val_pred = np.reshape(y_val_pred, (-1, window_size, window_size))
        y_val_pred = np.where(y_val == -1, -1, y_val_pred)
        del x_val
        y_score_gt = y_flat[y_flat != -1]
        y_val_pred_flat = y_val_pred.reshape(-1, 1)
        y_score_pred = y_val_pred_flat[y_flat != -1]
        if len(y_score_gt) >0:
            MSE += metrics.mean_squared_error(y_true=y_score_gt, y_pred=y_score_pred, squared=True)
        y_predict = np.append(y_predict, y_val_pred, axis=0)
        if count == 1:
            np.savez(f"batch{count}_test.npz", pred=y_predict, gt=y_val)
            # with open("first_batch_test.npy", "wb") as f:
            #     np.save(f, y_val)
            #     np.save(f, y_predict)
            print("save first batch test")

        del y_val_pred, y_val
    RMSE = math.sqrt(MSE / count)
    return RMSE, y_predict


if __name__ == "__main__":
    num_features = 8
    regressor = "SGD"
    # imageIdx = 1
    WINDOWSIZE = 16
    for imageIdx in range(1,2):
        test_data = f"data/data_test_c{num_features}_p{imageIdx}_new.hdf5"
        ckp_path = f"checkpoints{num_features}_unstand_filter0/PartialFit{regressor}_last.pkl"
        pred_res_path = f"result/update/c{num_features}_unstand_filter0/"
        if not os.path.exists(pred_res_path):
            os.mkdir(pred_res_path)

        BATCH_SIZE = 200*14*14
        test_dset = SatelliteSet(test_data,
                                 feature_windows=471969,
                                 windowsize=WINDOWSIZE,
                                 num_feature=num_features,
                                 multiple=False)
        test_loader = torch.utils.data.DataLoader(test_dset,
                                                  batch_size=BATCH_SIZE,
                                                  num_workers=1,
                                                  shuffle=False)
        # single file

        MODEL_TITLE = ckp_path.split("/")[-1].split(".")[0]+"-"+test_data.split("/")[-1].split(".")[0]

        print(MODEL_TITLE)
        with open(ckp_path, "rb") as file:
            reg = pickle.load(file)
        print("Finished model loading.")
        start = time.time()
        if regressor[:3] == "XGB":
            rmse, y_pred = do_validation_xgb(reg, test_loader, WINDOWSIZE)
        else:
            rmse, y_pred = do_validation(reg, test_loader, WINDOWSIZE)
        f = h5py.File(os.path.join(pred_res_path, MODEL_TITLE + ".hdf5"), "w")
        pred_set = f.create_dataset("prediction_dataset", data=y_pred)
        f.close()
        print(f"Time used: {time.time() - start}")
        print(f"Final validation score: \nrmse - {rmse}")


