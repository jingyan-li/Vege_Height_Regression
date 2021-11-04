import h5py
import numpy as np
from tqdm import tqdm

def remove_cloud(cloud, rgb, nir, CLOUD_THRESHOLD = 0):
    '''
    Remove cloud and average over rotations
    :param cloud:
    :param rgb:
    :param nir:
    :param CLOUD_THRESHOLD: preserve records less than threshold, otherwise mask out
    :return:
    '''
    no_cld_freq = np.zeros((cloud.shape[1], cloud.shape[2]))
    for i in range(cloud.shape[0]):
        no_cld_freq += cloud[i] <= CLOUD_THRESHOLD

    rgb_merge = np.zeros(rgb[0].shape)
    for i in range(rgb.shape[0]):
        rgb_merge += np.where((cloud[i] <= CLOUD_THRESHOLD)[:, :, np.newaxis], rgb[i], 0)
    rgb_avg = np.where(no_cld_freq[:, :, np.newaxis] > 0, rgb_merge / no_cld_freq[:, :, np.newaxis], -1)

    nir_merge = np.zeros(nir[0].shape)
    for i in range(nir.shape[0]):
        nir_merge += np.where(cloud[i] <= CLOUD_THRESHOLD, nir[i], 0)
    nir_avg = np.where(no_cld_freq > 0, nir_merge / no_cld_freq, -1)

    return rgb_avg, nir_avg


if __name__ == "__main__":

    dset = h5py.File("../data/dataset_rgb_nir_train.hdf5","r")
    CLOUDNAME = "CLD_"
    LAYERNAMES = ["INPT_","NIR_"]

    # Copy GT to h5py
    with h5py.File("../data/stacked_data_train.hdf5","w") as f:
        if "GT" not in f.keys():
            _ = f.create_dataset(f"GT", data=dset["GT"])
            del _

    # Apply cloud masks and stack 20 rotations into 1
    for i in tqdm(range(1, 5)):
        with h5py.File("../data/stacked_data_train.hdf5", "a") as f:
            rgb_layer_name = f"{LAYERNAMES[0]}{i}"
            nir_layer_name = f"{LAYERNAMES[1]}{i}"
            if (rgb_layer_name not in f.keys()) or (nir_layer_name not in f.keys()):
                cloud = dset[f"{CLOUDNAME}{i}"]
                rgb = dset[rgb_layer_name]
                nir = dset[nir_layer_name]

                rgb_avg, nir_avg = remove_cloud(cloud, rgb, nir)
                del cloud, rgb, nir

                # Save into h5py
                _ = f.create_dataset(rgb_layer_name, data=rgb_avg)
                del _
                _ = f.create_dataset(nir_layer_name, data=nir_avg)
                del _

