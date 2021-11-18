import torch
import torch.nn as nn
import numpy as np
from dataset_windows import SatelliteSet, flatten_batch_data, standardize_data

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch.nn.functional as F
from tqdm import tqdm
import h5py
from PIL import Image
import matplotlib.pyplot as plt


# def createDeepLabv3(outputchannels=1):
#     """DeepLabv3 class with custom head
#     Args:
#         outputchannels (int, optional): The number of output channels in your dataset masks. Defaults to 1.
#     Returns:
#         model: Returns the DeepLabv3 model with the ResNet101 backbone.
#     """
#     model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
#     model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
#     model.classifier = DeepLabHead(2048, outputchannels)
#     # Set the model in training mode
#     model.eval()
#     return model
#
#
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#
#         image_modules = list(createDeepLabv3(32).children())[:-1]
#         self.layer1 = nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1)
#         self.model = nn.Sequential(*image_modules)
#
#     def forward(self, tensor):
#         a = self.layer1(tensor)
#         # a = torch.tensor(a, dtype=torch.float64)
#         a = self.model(a.double())
#
#         return a

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

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class ASSP(nn.Module):

    def __init__(self, in_channels, out_channels=256):
        super(ASSP, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=6,
                               dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=12,
                               dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=18,
                               dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.conv5 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                               dilation=1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.convf = nn.Conv2d(in_channels=out_channels * 5, out_channels=out_channels, kernel_size=1, stride=1,
                               padding=0, dilation=1, bias=False)
        self.bnf = nn.BatchNorm2d(out_channels)
        self.adapool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)
        x5 = self.adapool(x)
        x5 = self.conv5(x5)
        x5 = self.bn5(x5)
        x5 = self.relu(x5)
        x5 = F.interpolate(x5, size=tuple(x4.shape[-2:]), mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # channels first
        x = self.convf(x)
        x = self.bnf(x)
        x = self.relu(x)
        return x


class Deeplabv3Resnet101(nn.Module):
    """Consturct Deeplabv3_Resnet101"""

    def __init__(self, nc=2, input_channel=4):
        super(Deeplabv3Resnet101, self).__init__()
        self.nc = nc
        self.backbone = ResNet_101(input_channel)
        self.assp = ASSP(in_channels=1024)
        self.out1 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1), nn.ReLU())
        self.dropout1 = nn.Dropout(0.5)
        self.up4 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=2)

        self.conv1x1 = nn.Sequential(nn.Conv2d(1024, 256, 1, bias=False), nn.ReLU())
        self.conv3x3 = nn.Sequential(nn.Conv2d(512, self.nc, 1), nn.ReLU())
        self.dec_conv = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())

    def forward(self, x):
        x = self.backbone(x)
        out1 = self.assp(x)
        out1 = self.out1(out1)
        out1 = self.dropout1(out1)
        out1 = self.up4(out1)
        # print(out1.shape)

        dec = self.conv1x1(x)
        dec = self.dec_conv(dec)
        dec = self.up4(dec)
        concat = torch.cat((out1, dec), dim=1)
        out = self.conv3x3(concat)
        out = self.up4(out)
        return out





if __name__ == "__main__":
    # input_tensor = torch.rand(4, 4, 128, 128)  # batch_size,input_channel,input_h,input_w
    # print(input_tensor)
    # model = Deeplabv3Resnet101(nc=32, input_channel=4)
    # out = model(input_tensor)
    # print(out.shape)

    # Parameters for loading data
    TRAIN_PATH = '../data/data_test_rgbReduced_delBlankRotations_Standardized.hdf5'
    IMAGE_NUM = 2
    WINDOWSIZE = 224
    # Sampling data
    SAMPLE = True
    SAMPLESIZE = 0.5
    # Batch size for train dataloader
    BATCH_SIZE = 128
    # Output features
    OUTPUT_FEATURES = 32

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
        sampling = list(range(1*length, 2*length))
        train_dset = torch.utils.data.Subset(train_dset, sampling)
    print(f"TRAIN dataset contains {len(train_dset)} windows")

    # Create dataloader
    train_loader = torch.utils.data.DataLoader(train_dset,
                                               batch_size=BATCH_SIZE,
                                               num_workers=0,
                                               shuffle=False)

    # Initialize models
    print("Loading model...")
    model = Deeplabv3Resnet101(nc=OUTPUT_FEATURES, input_channel=4)

    with h5py.File("P:\pf\pfstud\II_jingyli\data_test_features_c32_pic2.hdf5", "a") as f:
        x_features = np.zeros((len(train_dset), OUTPUT_FEATURES, WINDOWSIZE, WINDOWSIZE))
        y_gt = np.zeros((len(train_dset), WINDOWSIZE, WINDOWSIZE))
        i = 0
        for x, y in tqdm(train_loader):
            print(x.shape)
            print(y.shape)
            x_out = model(x)
            del x
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




    # img = Image.open('../1.jpg')
    #
    # import torchvision.transforms as T
    #
    # trf = T.Compose([T.Resize(256),
    #                  T.CenterCrop(224),
    #                  T.ToTensor()])
    # inp = trf(img).unsqueeze(0)
    # print(inp.shape)
    #
    # model = createDeepLabv3(24)
    # out = model(inp)["out"]
    # print(out.shape)





