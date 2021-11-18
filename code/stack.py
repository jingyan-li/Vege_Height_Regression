import h5py
import numpy as np

raw = h5py.File(f"P:\pf\pfstud\II_jingyli\data_train_rgbReduced_delBlankRotations_Standardized.hdf5", "r")
print(raw.keys())

for i in range(3, 4):

    print("start:", i)
    p1_rgb = raw["INPT"][i]
    print(p1_rgb.shape)
    p1_rgb = np.pad(p1_rgb, [[0, 220], [0, 220], [0, 0]])
    print(p1_rgb.shape)
    p1_rgb = np.reshape(p1_rgb, (2500, 3, 224, 224))
    print(p1_rgb.shape)

    p1_nir = raw["NIR"][i]
    print(p1_nir.shape)
    p1_nir = np.pad(p1_nir, [[0, 220], [0, 220]])
    print(p1_nir.shape)
    p1_nir = np.reshape(p1_nir, (2500, 1, 224, 224))
    print(p1_nir.shape)

    p116 = h5py.File(f"P:\pf\pfstud\II_jingyli\data_features_c32_pic{i+1}.hdf5", "r")
    print(p116.keys())

    p1_features = p116["Features"]
    print(p1_features.shape)
    # p1_features = np.reshape(p1_features, (11200, 11200, 16))
    # print(p1_features.shape)

    p1_gt = p116["GT"]
    print(p1_gt.shape)

    p1_f20 = np.concatenate([p1_features, p1_rgb, p1_nir], axis=1)
    print(p1_f20.shape)
    # p1_f20 = np.reshape(p1_f20, (2500, 20, 224, 224))

    with h5py.File(f"P:\pf\pfstud\II_jingyli\data_features_c36_pic{i+1}.hdf5", "a") as f:
        _ = f.create_dataset("Features", data=p1_f20)
        del _
        _ = f.create_dataset("GT", data=p1_gt)
        del _

