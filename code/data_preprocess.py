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
    LAYERNAMES = ["INPT","NIR"]

    # Copy GT to h5py
    with h5py.File("../data/data_train_rgbReduced.hdf5","w") as f:
        if "GT" not in f.keys():
            _ = f.create_dataset(f"GT", data=dset["GT"])
            del _

    # Apply cloud masks and stack 20 rotations into 1
    with h5py.File("../data/data_train_rgbReduced.hdf5", "a") as f:
        if (LAYERNAMES[0] not in f.keys()) or (LAYERNAMES[1] not in f.keys()):
            rgb_avg_all = np.zeros(np.concatenate((dset["GT"].shape, [3])))
            nir_avg_all = np.zeros(dset["GT"].shape)
            for i in tqdm(range(1, 5)):
                rgb_layer_name = f"{LAYERNAMES[0]}_{i}"
                nir_layer_name = f"{LAYERNAMES[1]}_{i}"
                cloud = dset[f"{CLOUDNAME}{i}"]
                rgb = dset[rgb_layer_name]
                nir = dset[nir_layer_name]

                rgb_avg, nir_avg = remove_cloud(cloud, rgb, nir)
                del cloud, rgb, nir
                rgb_avg_all[i-1] = rgb_avg
                nir_avg_all[i-1] = nir_avg
            # Save into h5py
            _ = f.create_dataset(LAYERNAMES[0], data=rgb_avg_all)
            del _
            _ = f.create_dataset(LAYERNAMES[1], data=nir_avg_all)
            del _
