import h5py
import numpy as np
from tqdm import tqdm


def normalize(af_inpt):
    # Mask out clouds
    no_data_mask = af_inpt <= -1.0
    af_inpt_ma = np.ma.array(af_inpt, mask=no_data_mask)
    # Standardization to mean = 0, std = 1
    miu = np.mean(af_inpt_ma, axis=(0, 1))
    sigma = np.std(af_inpt_ma, axis=(0, 1))
    norm_af_inpt_ma = (af_inpt_ma - miu) / sigma
    # Normalization to [0,1]
    maxi = np.max(norm_af_inpt_ma, axis=(0, 1))
    mini = np.min(norm_af_inpt_ma, axis=(0, 1))
    sta_af_inpt_ma = (norm_af_inpt_ma - mini) / (maxi - mini)
    return sta_af_inpt_ma


if __name__=="__main__":
    af_dset = h5py.File("../data/data_train_rgbReduced_delBlankRotations.hdf5", "r")
    print(af_dset.keys())
    rgb_avg_all = np.zeros(af_dset["INPT"].shape)
    nir_avg_all = np.zeros(af_dset["NIR"].shape)
    for i in tqdm(range(4)):
        rgb_avg_all[i] = normalize(af_dset["INPT"][i]).data
        nir_avg_all[i] = normalize(af_dset["NIR"][i]).data
    print("Writing to file...")
    # Save to h5py
    with h5py.File("../data/data_train_rgbReduced_delBlankRotations_Standardized.hdf5", "a") as f:
        # Save into h5py
        _ = f.create_dataset("INPT", data=rgb_avg_all)
        del _
        _ = f.create_dataset("NIR", data=nir_avg_all)
        del _
        _ = f.create_dataset("GT", data=af_dset["GT"])
        del _