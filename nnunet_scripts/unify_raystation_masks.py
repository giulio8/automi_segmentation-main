import nibabel as nib
import os
import numpy as np
from pathlib import Path
from map_to_binary import class_map

def combine_raystation_masks_into_one(masks_folder_path, masks_list, masks_result_path, mask_name):

    ref_file_name = f'{masks_list[0]}.nii.gz'
    ref_img = nib.load(os.path.join(masks_folder_path, ref_file_name))
    img_out = np.zeros(ref_img.shape).astype(np.uint8)
    final_mask_file_name = f'{mask_name}.nii.gz'

    for idx, mask in enumerate(masks_list):

        mask_file_name = f'{mask}.nii.gz'
        mask_dir = os.path.join(masks_folder_path, mask_file_name)
        img = nib.load(mask_dir).get_fdata()
        img_out[img > 0.5] = 1

    nib.save(nib.Nifti1Image(img_out, ref_img.affine), os.path.join(masks_result_path, final_mask_file_name))

def combine_masks_patients(dataset_path, mask_list, mask_name):
    for patient in os.listdir(dataset_path):
        patient_dir_path = os.path.join(dataset_path, patient)
        combine_raystation_masks_into_one(patient_dir_path, mask_list, patient_dir_path, mask_name)
        print(f"Created multilabel \'{mask_name}\' file for patient {patient}!"
              f"\n Patients left: {len(os.listdir(dataset_path)) - os.listdir(dataset_path).index(patient) - 1}")


if __name__ == "__main__":
    dataset_path = '/mnt/storage/ricardo/nnUnetAUTOMI/dataset/nnUnet_raw/nifti_ct_multiple_inputs/'
    mask_list = ['mask_Kidney_L', 'mask_Kidney_R']
    mask_name = 'mask_kidneys'
    combine_masks_patients(dataset_path, mask_list, mask_name)