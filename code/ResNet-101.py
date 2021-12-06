import torch
import torch.nn as nn
import numpy as np
from dataset_windows import SatelliteSet, flatten_batch_data, standardize_data

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch.nn.functional as F
from tqdm import tqdm
import h5py


class ResNet_101(nn.Module):
    """Load pretrained model resnet101"""

    def __init__(self, in_channels=4, conv1_out=64):
        super(ResNet_101, self).__init__()
        backbone = models.resnet101(pretrained=True)

        # Change the input channels from 3 to 4 and set the same weight for NIR as R
        pretrained_conv1 = backbone.conv1.weight.clone()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight[:, :3] = pretrained_conv1
        self.conv1.weight[:, -1] = pretrained_conv1[:, 0]

        self.bn1 = backbone.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool  = backbone.avgpool

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)

        return x


if __name__ == "__main__":
    # input_tensor = torch.rand(4, 4, 128, 128)  # batch_size,input_channel,input_h,input_w
    # print(input_tensor)
    # model = Deeplabv3Resnet101(nc=32, input_channel=4)
    # out = model(input_tensor)
    # print(out.shape)

    # Parameters for loading data
    TRAIN_PATH = 'P:\pf\pfstud\II_jingyli\data_test_rgbReduced_delBlankRotations.hdf5'
    IMAGE_NUM = 2
    WINDOWSIZE = 16
    # Sampling data
    SAMPLE = True
    SAMPLESIZE = 0.5
    # Batch size for train dataloader
    BATCH_SIZE = 128
    # Output features
    OUTPUT_FEATURES = 16

    for j in range(2):
        print("Start:", j)
        # Set seed, so that results can be reproduced
        np.random.seed(2021)

        # Loading data
        print("Loading data...")
        train_dset = SatelliteSet(TRAIN_PATH, IMAGE_NUM, WINDOWSIZE)
        print(f"Original dataset contains {len(train_dset)} samples")

        # Sampling
        if SAMPLE:
            print("Sampling...")
            #sampling = np.random.randint(len(train_dset), size=round(len(train_dset) * SAMPLESIZE)
            length = round(len(train_dset) * SAMPLESIZE)
            sampling = list(range(j*length, (j+1)*length))
            train_dset = torch.utils.data.Subset(train_dset, sampling)
        print(f"TRAIN dataset contains {len(train_dset)} windows")

        # Create dataloader
        train_loader = torch.utils.data.DataLoader(train_dset,
                                                   batch_size=BATCH_SIZE,
                                                   num_workers=0,
                                                   shuffle=False)

        # Initialize models
        print("Loading model...")
        # model = Deeplabv3Resnet101(nc=OUTPUT_FEATURES, input_channel=4)
        #model = createDeepLabv3(16)
        model = ResNet_101()
        model.eval()

        with h5py.File(f"P:\pf\pfstud\II_jingyli\data_test_c8_p{j+1}_new.hdf5", "a") as f:
            x_features = np.zeros((len(train_dset), 8, 16, 16))
            y_gt = np.zeros((len(train_dset), WINDOWSIZE, WINDOWSIZE))
            i = 0
            for x, y in tqdm(train_loader):
                print(x.shape)
                print(y.shape)
                x_out = model(x)
                x_out = x_out.reshape(x.shape[0],8,16,16)
                print(x_out.shape)
                x_out = x_out.detach().numpy()
                x_features[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = x_out
                del x_out
                y = y.detach().numpy()
                y_gt[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = y
                del y
                i += 1

            _ = f.create_dataset("Features", data=x_features)
            del _
            _ = f.create_dataset("GT", data=y_gt)
            del _