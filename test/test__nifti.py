'''
This script is used to evaluate if nifti files are in the correct value range and shape and so on.
'''

import os, sys
import nibabel as nib
import numpy as np


def check_nifti_file(nifti_path):
    # Loads the nifti file
    nii_img = nib.load(nifti_path)
    nii_array = nii_img.get_fdata()

    # Print name of the nifti file
    print(f'Name: {os.path.split(nifti_path)[-1]}')
    # Print the shape of the nifti file
    print(f'Shape: {nii_array.shape}')
    # Print the value range of the nifti file
    print(f'Value range: {np.min(nii_array)} - {np.max(nii_array)}')


if __name__ == "__main__":
    nifti_path1 = '/home/ricardo/Desktop/groundtruth/AUTOMI_00006.nii.gz'
    nifti_path2 = '/home/ricardo/Desktop/bone_subtraction_results/AUTOMI_00006.nii.gz'
    nifti_path3 = '/home/ricardo/Desktop/bones/AUTOMI_00006.nii.gz'

    check_nifti_file(nifti_path1)
    check_nifti_file(nifti_path2)
    check_nifti_file(nifti_path3)