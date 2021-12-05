import math

import h5py
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


def batch_data_preprocess(x, y):
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


def do_validation_xgb(model, dataloader):
    # Validation score
    MSE = 0
    count = 0
    y_predict = np.empty((0, 224, 224))
    for x_val, y_val in dataloader:
        count += 1
        x_scaled, y_flat = batch_data_preprocess(x_val, y_val)
        y_val_pred = model.predict(xgb.DMatrix(x_scaled))
        y_val_pred = np.reshape(y_val_pred, (-1, 224, 224))
        y_val_pred = y_val_pred[np.where(y_val==-1,-1,y_val_pred)]
        del x_val
        y_score_gt = y_flat[y_flat != -1]
        y_score_pred = y_val_pred[y_flat != -1]
        MSE += metrics.mean_squared_error(y_true=y_score_gt, y_pred=y_score_pred, squared=False)
        y_predict = np.append(y_predict, y_val_pred)
        if count == 1:
            with open("first_batch_test.npy", "wb") as f:
                np.save(f, y_val_pred)
                np.save(f, y_predict)
            print("save first batch test")
        else:
            return
        del y_val_pred, y_val
    RMSE = math.sqrt(MSE/count)
    return RMSE, y_predict


def do_validation(model, dataloader):
    # Validation score
    MSE = 0
    count = 0
    y_predict = np.empty((0, 224, 224))
    for x_val, y_val in dataloader:
        count += 1
        x_val, y_val = batch_data_preprocess(x_val, y_val)
        y_val_pred = model.predict(x_val)
        y_val_pred = np.reshape(y_val_pred, (-1, 224, 224))
        del x_val
        MSE += metrics.mean_squared_error(y_true=y_val, y_pred=y_val_pred)
        y_predict = np.append(y_predict, y_val_pred)
        del y_val_pred, y_val
    RMSE = math.sqrt(MSE/count)
    return RMSE, y_predict


if __name__ == "__main__":
    test_data = r"D:\II_LAB2_DATA\data_test_features_c16_pic1.hdf5"
    ckp_path = r""
    pred_res_path = r""
    num_features = 16
    BATCH_SIZE = 200
    test_dset = SatelliteSet(test_data,
                             num_feature=num_features,
                             multiple=False)
    test_loader = torch.utils.data.DataLoader(test_dset,
                                              batch_size=BATCH_SIZE,
                                              num_workers=1,
                                              shuffle=False)

    # multiple file
    """
    for ckp in os.listdir(ckp_path):
        ckp_path = os.path.join(ckp_path, ckp)
        MODEL_TITLE = os.path.splitext(ckp)[0]
        print(MODEL_TITLE)
        with open(ckp_path, "rb") as file:
            reg = pickle.load(file)
        print("Finished model loading.")
        start = time.time()
        if MODEL_TITLE[:3] == "XGB":
            rmse,y_pred = do_validation_xgb(reg, test_loader)
        else:
            rmse,y_pred = do_validation(reg, test_loader)
        f = h5py.File(os.path.join(pred_res_path, MODEL_TITLE+".hdf5"),"w")
        pred_set = f.create_dataset("prediction_dataset", data=y_pred)
        f.close()
        print(f"Time used: {time.time() - start}")
        print(f"Final validation score: \nrmse - {rmse}")

     """

    # single file
    ckp_path = r"D:\yurjia\ImageInterpretation_Regression\code\checkpoints16\XGB_checkpoint.pkl"
    MODEL_TITLE = "XGB_16"
    print(MODEL_TITLE)
    with open(ckp_path, "rb") as file:
        reg = pickle.load(file)
    print("Finished model loading.")
    start = time.time()
    if MODEL_TITLE[:3] == "XGB":
        rmse, y_pred = do_validation_xgb(reg, test_loader)
    else:
        rmse, y_pred = do_validation(reg, test_loader)
    f = h5py.File(os.path.join(pred_res_path, MODEL_TITLE + ".hdf5"), "w")
    pred_set = f.create_dataset("prediction_dataset", data=y_pred)
    f.close()
    print(f"Time used: {time.time() - start}")
    print(f"Final validation score: \nrmse - {rmse}")
