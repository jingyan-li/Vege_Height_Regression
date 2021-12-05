from torchvision.datasets.vision import VisionDataset
import h5py
import numpy as np
from sklearn import preprocessing
from skimage.feature import local_binary_pattern,greycomatrix,greycoprops
from skimage.util import view_as_windows
from skimage.color import rgb2gray

class SatelliteSet(VisionDataset):

    def __init__(self, dfile_path, image_num, windowsize=128, split="train"):
        assert split in ["train", "validation"], "The split parameter accepts train or validation only"
        # Set split
        self.split = split

        self.wsize = windowsize
        super().__init__(None)

        self.sh_x, self.sh_y = 10980,10980 # size of each image
        self.num_smpls = image_num

        self.pad_x = (self.sh_x - (self.sh_x % self.wsize))
        self.pad_y = (self.sh_y - (self.sh_y % self.wsize))
        self.sh_x = self.pad_x + self.wsize
        self.sh_y = self.pad_y + self.wsize
        self.num_windows = self.num_smpls * self.sh_x / self.wsize * self.sh_y / self.wsize
        self.num_windows = int(self.num_windows)
        self.has_data = False

        self.file_path = dfile_path

    # ugly fix for working with windows
    # Windows cannot pass the h5 file to sub-processes, so each process must access the file itself.
    def load_data(self):
        h5 = h5py.File(self.file_path, 'r')

        self.RGB = h5["RGB"]
        self.NIR = h5["NIR"]
        self.CLD = h5["CLD"]
        self.GT = h5["GT"]

        self.has_data = True

    def __getitem__(self, index):
        if not self.has_data:
            self.load_data()

        """Returns a data sample from the dataset.
        """
        # determine where to crop a window from all images (no overlap)
        m = index * self.wsize % self.sh_x  # iterate from left to right
        # increase row by windows size everytime m increases
        n = (int(np.floor(index * self.wsize / self.sh_x)) * self.wsize) % self.sh_x
        # determine which batch to use
        b = (index * self.wsize * self.wsize // (self.sh_x * self.sh_y)) % self.num_smpls

        # crop all data at the previously determined position
        RGB_sample = self.RGB[b, n:n + self.wsize, m:m + self.wsize]
        NIR_sample = self.NIR[b, n:n + self.wsize, m:m + self.wsize]
        CLD_sample = self.CLD[b, n:n + self.wsize, m:m + self.wsize]
        GT_sample = self.GT[b, n:n + self.wsize, m:m + self.wsize]

        # normalize NIR and RGB by maximumg possible value
        NIR_sample = np.asarray(NIR_sample, np.float32) / (2 ** 16 - 1)
        RGB_sample = np.asarray(RGB_sample, np.float32) / (2 ** 8 - 1)
        R = RGB_sample[:, :, 0]
        G = RGB_sample[:, :, 1]
        B = RGB_sample[:, :, 2]
        # Pseudo zero (to avoid divide by zero)
        eps = 10e-10
        # Add other features
        NDVI = (NIR_sample - R) / np.where(NIR_sample + R == 0, NIR_sample+R+eps, NIR_sample+R)
        MSAVI = (1 / 2) * (2 * (NIR_sample+1) - np.sqrt(
                    (2 * NIR_sample + 1) ** 2 - 8 * (NIR_sample - R)))
        VARI = (G-R) / np.where(G+R-B == 0, G+R-B+eps, G+R-B)

        CI1 = 3 * NIR_sample / np.where(R+G+B == 0, R+G+B+eps, R+G+B)
        CI2 = (NIR_sample + R + G + B) / 4
        LBP = detect_texture(RGB_sample)
        # GLCM = glcm_by_window(RGB_sample)

        X_sample = np.concatenate([RGB_sample,
                                   np.expand_dims(NIR_sample, axis=-1),
                                   np.expand_dims(NDVI, axis=-1),
                                   np.expand_dims(MSAVI, axis=-1),
                                   np.expand_dims(VARI, axis=-1),
                                   np.expand_dims(CI1, axis=-1),
                                   np.expand_dims(CI2, axis=-1),
                                   np.expand_dims(LBP, axis=-1),
                                   # GLCM
                                   ], axis=-1)

        ### correct gt data ###
        # first assign gt at the positions of clouds
        cloud_positions = np.where(CLD_sample > 10)
        GT_sample[cloud_positions] = 2
        # second remove gt where no data is available - where the max of the input channel is zero
        idx = np.where(np.max(X_sample, axis=-1) == 0)  # points where no data is available
        GT_sample[idx] = 99  # 99 marks the absence of a label and it should be ignored during training
        GT_sample = np.where(GT_sample > 3, 99, GT_sample)
        # pad the data if size does not match
        sh_x, sh_y = np.shape(GT_sample)
        pad_x, pad_y = 0, 0
        if sh_x < self.wsize:
            pad_x = self.wsize - sh_x
        if sh_y < self.wsize:
            pad_y = self.wsize - sh_y

        x_sample = np.pad(X_sample, [[0, pad_x], [0, pad_y], [0, 0]])
        gt_sample = np.pad(GT_sample, [[0, pad_x], [0, pad_y]], 'constant',
                           constant_values=[99])  # pad with 99 to mark absence of data

        # pytorch wants the data channel first - you might have to change this
        x_sample = np.transpose(x_sample, (2, 0, 1))
        return np.asarray(x_sample), gt_sample

    def __len__(self):
        return self.num_windows


def flatten_batch_data(x, y):
    '''
    Make an image into flat features N*M (N represents number of pixels, M denotes the number of features)
    :param x: data
    :param y: label
    :return:
    '''
    x = np.transpose(x.numpy(), [0, 2, 3, 1])  # Image windows, H, W, Features
    # Mask out y==99
    havedata_mask = y!=99
    # y = np.where(y == 99, 3, y)
    # print(x.shape)
    # print(y.shape)
    y = y[havedata_mask]
    x = x[havedata_mask]

    x_flat = x.reshape(-1, x.shape[-1])  # Flat as pixels, features
    y_flat = y.reshape(-1, 1)  # Flat as pixels, label

    return x_flat, y_flat


def standardize_data(x):
    '''
    Transform x into unit variance and zero mean
    :param x: input data
    :return:
    '''
    scaler = preprocessing.StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    return x_scaled


def detect_texture(x):
    '''
    get texture feature based on LBP
    :param x: input data (RGB sample)
    :return:
    '''
    gray_scale = rgb2gray(x)
    lbp = local_binary_pattern(gray_scale, 8, 1, method="default")
    return lbp


def spatial_feature(x):
    '''
    get spatial information
    :param x:
    :return:
    '''


def glcm_by_window(x):
    '''
    Calculate texture information
    :param x: RGB of an image [H,W,3]
    :return:
    '''
    glcm_result = np.zeros((x.shape[0], x.shape[1], 4*4))
    WINDOW_SIZE = 9
    padding = WINDOW_SIZE//2
    # Pad the image
    pad_x = np.pad(x, pad_width=((padding, padding), (padding, padding), (0, 0)), mode="symmetric")
    # View as windows
    windows = view_as_windows(pad_x, window_shape=(WINDOW_SIZE,WINDOW_SIZE,pad_x.shape[-1]), step=1)
    # Calculate GLCM for each window
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            # Get a small window
            glcm_result[i, j] = 0 #GLCM(np.squeeze(windows[i, j], 0))
    # Return stacked results
    return glcm_result


def GLCM(x):
    '''
    Grey Level Cooccurence Matrix (contrast, homogeneity, correlation and disimilarity of four directions are calculated)
    :param x: RGB of a window from an image
    :return: array of length 16
    '''
    gray_scale = rgb2gray(x*255)
    g = greycomatrix(gray_scale.astype(int), distances=[5], angles=[0,np.pi/2,np.pi,np.pi/2*3], levels=256)
    disimilarity = greycoprops(g, "dissimilarity")[0]
    contrast = greycoprops(g, "contrast")[0]
    homogeneity = greycoprops(g, "homogeneity")[0]
    correlation = greycoprops(g, "correlation")[0]
    glcm = np.concatenate((contrast, homogeneity, correlation, disimilarity), axis=None)
    return glcm

# if __name__ == "__main__":
#     colormap = [[47, 79, 79], [0, 255, 0], [255, 255, 255], [0, 0, 0]]
#     colormap = np.asarray(colormap)
#     import matplotlib.pyplot as plt
#     from tqdm import tqdm
#     import torch
#
#     dset = SatelliteSet(dfile_path="../data/dataset_test.h5", windowsize = 128)
#     # create dataloader that samples batches from the dataset
#     train_loader = torch.utils.data.DataLoader(dset,
#                                                batch_size=8,
#                                                num_workers=1,
#                                                shuffle=False)
#
#     # Please note that random shuffling (shuffle=True) -> random access.
#     # this is slower than sequential reading (shuffle=False)
#     # If you want to speed up the read performance but keep the data shuffled, you can reshape the data to a fixed window size
#     # e.g. (-1,4,128,128) and shuffle once along the first dimension. Then read the data sequentially.
#     # another option is to read the data into the main memory h5 = h5py.File(root, 'r', driver="core")
#
#     # plot some examples
#     f, axarr = plt.subplots(ncols=3, nrows=8)
#
#     for x, y in tqdm(train_loader):
#         x = np.transpose(x, [0, 2, 3, 1])
#         y = np.where(y == 99, 3, y)
#         for i in range(len(x)):
#             axarr[i, 0].imshow(x[i, :, :, :3])
#             axarr[i, 1].imshow(x[i, :, :, -1])
#             axarr[i, 2].imshow(colormap[y[i]] / 255)
#
#         plt.show()
#         quit()
