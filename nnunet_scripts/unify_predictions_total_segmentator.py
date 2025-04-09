# Script to turn the predictions of the nnUNet into a single file for each patient
import nibabel as nib
import os
import numpy as np
from pathlib import Path
from map_to_binary import class_map

def combine_masks_to_multilabel_file(masks_dir, multilabel_file):
    """
    Generate one multilabel nifti file from a directory of single binary masks of each class.
    This multilabel file is needed to train a nnU-Net.

    masks_dir: path to directory containing all the masks for one subject
    multilabel_file: path of the output file (a nifti file)
    """
    ref_img = nib.load(os.path.join(masks_dir,"clavicula_left.nii.gz"))
    img_out = np.zeros(ref_img.shape).astype(np.uint8)
    masks = os.listdir(masks_dir)

    for idx, mask in enumerate(masks):
        mask_dir = os.path.join(masks_dir, mask)
        img = nib.load(mask_dir).get_fdata()
        img_out[img > 0.5] = 1

    nib.save(nib.Nifti1Image(img_out, ref_img.affine), multilabel_file)


if __name__ == "__main__":
    total_segmentator_output_dir = "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset001_AUTOMI100/total_segmentator_bone_segmentations/"
    for patient_dir in os.listdir(total_segmentator_output_dir):
        patient_dir_path = os.path.join(total_segmentator_output_dir, patient_dir)
        if not os.path.isfile(patient_dir_path):
            multilabel_file = os.path.join(total_segmentator_output_dir, f"{patient_dir}_multilabel.nii.gz")
            if not os.path.exists(multilabel_file):
                combine_masks_to_multilabel_file(patient_dir_path, multilabel_file)
                print(f"Created multilabel file for patient {patient_dir}!"
                      f"\n Patients left: {len(os.listdir(total_segmentator_output_dir)) - os.listdir(total_segmentator_output_dir).index(patient_dir) - 1}")