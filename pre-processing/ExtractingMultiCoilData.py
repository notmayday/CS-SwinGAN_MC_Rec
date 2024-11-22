import numpy as np
import glob
import nibabel as nib
import os
import random
import h5py
import shutil
from multiprocessing import Pool
from PIL import Image
import matplotlib.pyplot as plt
import pickle

def ifft2(kspace_cplx):
    return np.fft.ifftshift(np.fft.ifft2(kspace_cplx),axes=[2,3])

def fft2(img):
    return np.fft.fft2(np.fft.fftshift(img,axes=[2,3]))

directory_path='/media/samuel/storage/UIH/TEMP/data/h5/200/'
save_path='/media/samuel/storage/UIH/V3+/data/h5/mc/'
h5_files = [file for file in os.listdir(directory_path) if file.endswith('.h5')]
size=np.size(h5_files)
new_size=320
for file_name in h5_files:
    file_path = os.path.join(directory_path, file_name)
    with h5py.File(file_path, 'r') as h5_file:
        print(f"Processing {file_name}:")
        rss = h5_file['reconstruction_rss']
        rss= np.array(rss)
        data = rss
        data = np.array(data).transpose((1, 2, 0))
        rss_shape = rss.shape
        rss_hw = new_size
        ori_data_shape= np.shape(data)
        ori_size = [ori_data_shape[0],ori_data_shape[1]]
        kspace = h5_file['kspace']
        ori_kspace_shape = np.shape(kspace)
        [slice_number, coil_number, readout, phase]=ori_kspace_shape
        imgs = ifft2(kspace)
        cropped_imgs = imgs[:, :, int(0.5 * (readout - rss_hw)) : int(0.5 * (readout + rss_hw)) , int(0.5*(phase-rss_hw)): int(0.5*(phase+rss_hw))]
        cropped_kspaces = fft2(cropped_imgs)
        rss_imgs = np.sqrt(np.sum(np.abs(cropped_imgs) ** 2, axis=1))
        patient_name = os.path.split(file_path)[1].replace('h5', 'hdf5')
        output_file_path = save_path + patient_name
        with h5py.File(output_file_path, 'w') as f:
            dset1 = f.create_dataset('cropped_kspaces', cropped_kspaces.shape, data=cropped_kspaces, compression="gzip", compression_opts=9)
            dset2 = f.create_dataset('cropped_imgs', cropped_imgs.shape, data=cropped_imgs, compression="gzip", compression_opts=9)
            dset3 = f.create_dataset('rss_imgs', rss_imgs.shape, data=rss_imgs, compression="gzip", compression_opts=9)
