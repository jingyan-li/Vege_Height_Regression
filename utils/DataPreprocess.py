from sklearn.preprocessing import StandardScaler
import h5py
import numpy as np
import time
import os
from tqdm import tqdm
from torchvision.datasets.vision import VisionDataset
from sklearn import model_selection


def standardize_data(x):
    '''
    Transform x into unit variance and zero mean
    :param x: input data
    :return:
    '''
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    return x_scaled


def get_raw_feature_from_npfile(np_file):
    dset = h5py.File(np_file, "r")
    RGB_LayerName = "INPT_"
    NIR_LayerName = "NIR_"
    num_img = dset["GT"].shape[0]
    # x_train = np.array([])
    # y_train = np.array([])
    x_train = np.empty((0, 4), float)
    y_train = np.empty((0, 1), float)
    for i in tqdm(range(num_img)):
        rgb_lyr = RGB_LayerName + str(i + 1)
        nir_lyr = NIR_LayerName + str(i + 1)
        nir = dset[nir_lyr][:]
        GT = dset["GT"][i]
        GT = np.where(nir == -1, -1, GT)

        NIR = nir[GT != -1]

        RGB = dset[rgb_lyr][:]
        RGB = RGB[GT != -1]

        GT = GT[GT != -1]

        trainx = np.concatenate([RGB, np.expand_dims(NIR, axis=-1)], axis=-1)
        x_train = np.append(x_train, trainx, axis=0)

        trainy = GT.reshape(-1, 1)
        y_train = np.append(y_train, trainy, axis=0)

    return x_train, y_train


def get_pretrained_feature_from_h5(h5_file, num_feature=16,multiple=False):
    if multiple:
        x_data = np.empty((0, num_feature), float)
        y_data = np.empty((0, 1), float)
        for file in os.listdir(h5_file):
            h5 = h5py.File(os.path.join(h5_file, file), 'r')

            fea = h5["Features"]
            gt = h5['GT']
            print("orginal h5 file feature shape:", fea.shape)
            print("orginal h5 file gt shape:", gt.shape)
            num_feat = fea.shape[1]

            fea = np.transpose(fea, (0, 2, 3, 1))
            x_sample = fea.reshape(-1, 1, num_feat)
            y_flat = np.asarray(gt).reshape(-1, 1)

            x_data_i = x_sample[np.where(y_flat != -1)]
            y_data_i = y_flat[y_flat != -1].reshape(-1, 1)
            _, x_data_i, _, y_data_i = model_selection.train_test_split(x_data_i,y_data_i,test_size=0.5, random_state=42)
            x_data = np.append(x_data, x_data_i, axis=0)
            y_data = np.append(y_data, y_data_i, axis=0)
            print("after process x feature shape:", x_data.shape)
            print("after process gt shape:", y_data.shape)
            print("Read {} samples with {} features.".format(x_data.shape[0], x_data.shape[1]))
    else:
        h5 = h5py.File(h5_file, 'r')

        fea = h5["Features"]
        gt = h5['GT']
        print("orginal h5 file feature shape:", fea.shape)
        print("orginal h5 file gt shape:", gt.shape)
        num_feat = fea.shape[1]

        fea = np.transpose(fea, (0, 2, 3, 1))
        x_sample = fea.reshape(-1, 1, num_feat)
        y_flat = np.asarray(gt).reshape(-1, 1)

        x_data = x_sample[np.where(y_flat != -1)]
        y_data = y_flat[y_flat != -1].reshape(-1, 1)
        print("after process x feature shape:", x_data.shape)
        print("after process gt shape:", y_data.shape)
        print("Read {} samples with {} features.".format(x_data.shape[0], x_data.shape[1]))
    return x_data, y_data


class SatelliteSet(VisionDataset):
    def __init__(self, dfile_path, windowsize=224, multiple=False, num_feature=16):
        super().__init__(None)
        self.wsize = windowsize
        self.file_path = dfile_path
        self.sh_x, self.sh_y = 224, 224  # size of each image
        self.feat_window = 2500
        self.multiple = multiple
        self.num_feature = num_feature
        if multiple:
            self.num_windows = 4 * self.feat_window
        else:
            self.num_windows = self.feat_window

    def __getitem__(self, index):
        if self.multiple:
            num_pic = index//self.feat_window
            # file = "D:\II_LAB2_DATA\c16\data_features_c16_pic"+ str(num_pic+1) + ".hdf5"
            file = "D:\II_LAB2_DATA\c" + str(self.num_feature) + "\data_features_c" + str(self.num_feature) +"_pic" + str(num_pic + 1) + ".hdf5"
            feat_set = h5py.File(file, 'r')
            features = feat_set["Features"]
            GT = feat_set["GT"]

            num_feat = features.shape[1]
            self.x_data = features[index % self.feat_window, :, :, :]
            self.y_data = GT[index % self.feat_window, :, :]
            #feat = np.transpose(features, (0, 2, 3, 1))

        else:
            feat_set = h5py.File(self.file_path, 'r')
            features = feat_set["Features"]
            GT = feat_set["GT"]
            num_feat = features.shape[1]
            # feat = np.transpose(self.features, (0, 2, 3, 1))
            self.x_data = features[index, :, :, :]
            self.y_data = GT[index, :, :]

        return self.x_data, self.y_data

    def __len__(self):
        return self.num_windows

