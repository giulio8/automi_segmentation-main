import matplotlib.pyplot as plt
import numpy as np

npz_file_path = '/mnt/storage/ricardo/nnUnetAUTOMI/dataset/nnUnet_raw/AUTOMI_00004.npz'

# Visualize a slice of the npz file
npz_file = np.load(npz_file_path)
npz = npz_file['data']
print("npz shape:", npz.shape)

slice= 100
plt.imshow(npz[1,slice, :, :])
plt.show()