import torch
import numpy as np
from dataset_windows import SatelliteSet, flatten_batch_data, standardize_data

from tensorflow.keras.applications import ResNet50V2

if __name__ == "__main__":
    # Parameters for loading data
    TRAIN_PATH = '../data/stacked_data_train.hdf5'
    IMAGE_NUM = 3
    WINDOWSIZE = 224
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
    model = models.resnet18(pretrained=True)

    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # Reshape the last Layer
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(fc_inputs, len())