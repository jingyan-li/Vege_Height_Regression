import torch
import torch.nn as nn
import numpy as np
from dataset_windows import SatelliteSet, flatten_batch_data, standardize_data

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch


def createDeepLabv3(outputchannels=1):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.eval()
    return model


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        image_modules = list(createDeepLabv3(32).children())[:-1]
        self.layer1 = nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1)
        self.model = nn.Sequential(*image_modules)

    def forward(self, tensor):
        a = self.layer1(tensor)
        # a = torch.tensor(a, dtype=torch.float64)
        a = self.model(a.double())

        return a


if __name__ == "__main__":
    # Parameters for loading data
    TRAIN_PATH = '../data/stacked_data_train.hdf5'
    IMAGE_NUM = 1
    WINDOWSIZE = 128
    # Sampling data
    SAMPLE = True
    SAMPLESIZE = 0.1
    # Batch size for train dataloader
    BATCH_SIZE = 256
    # K-Fold Classification
    KFOLD = 5


    # Set seed, so that results can be reproduced
    np.random.seed(2021)

    # Loading data
    print("Loading data...")
    train_dset = SatelliteSet(TRAIN_PATH, IMAGE_NUM, WINDOWSIZE)
    print(f"Original dataset contains {len(train_dset)} samples")

    # Sampling
    if SAMPLE:
        print("Sampling...")
        sampling = np.random.randint(len(train_dset), size=round(len(train_dset) * SAMPLESIZE))
        train_dset = torch.utils.data.Subset(train_dset, sampling)
    print(f"TRAIN dataset contains {len(train_dset)} windows")

    # Create dataloader
    train_loader = torch.utils.data.DataLoader(train_dset,
                                               batch_size=BATCH_SIZE,
                                               num_workers=0,
                                               shuffle=False)

    # Initialize models
    model = MyModel()

    for x, y in tqdm(train_loader):
        print(x.shape)
        x_out = model(x)["out"]
        print(x_out.shape)



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





