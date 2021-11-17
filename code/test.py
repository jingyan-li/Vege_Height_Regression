# from sklearn.neighbors import KNeighborsRegressor
import numpy as np

# X = np.random.randint(0, 100, size = [128*128, 64])
# y = np.random.randint(0, 10, size=[128*128])
#
# model = KNeighborsRegressor(n_neighbors=2)
#
# model.fit(X, y)

# import h5py
#
# dset = h5py.File("../data/data_train_features.hdf5", 'r')
# print(dset.keys())

# b = 16
# x = np.zeros((32, 224, 224))
# print(x[:, :3].shape)
# print(x.shape)
# for i in range(2):
#     x1 = np.random.randint(0, 10, size = (b, 224, 224))
#     print(x1.shape)
#     x[i*b : (i+1)*b] = x1
# print(x)

from torchvision import models
import torch.nn as nn

backbone = models.resnet101(pretrained=True)
pretrained_conv1 = backbone.conv1.weight.clone()
print(pretrained_conv1.shape)
conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
conv1.weight[:, :3] = pretrained_conv1
conv1.weight[:, -1] = pretrained_conv1[:, -1]
print(conv1.weight[:, 0].shape)