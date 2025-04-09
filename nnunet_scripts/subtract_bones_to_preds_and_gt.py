'''
 This script is used to prepare the .npz files of the predictions of the nnunet and of the groundtruth of the
original dataset. The goal of this experiment is to use the predictions of the bones created by the TotalSegmentator
and subtract them to the PTV total predictions. This way we can evaluate the PTV predictions without the bones, because
the bones are the easiest part to predict and we are trying to see improvements in the lymph nodes and other structures.
'''

import os, sys
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import shutil
import csv

def subtract_masks_folder(masks1_folder, masks2_folder, results_path, mask1_file_ending='.nii.gz', mask2_file_ending='.nii.gz'):
    '''
    This function will subtract the masks2 from the masks1 and save the results in the results_path.
    The masks are expected to be in .ni.gz format.
    :param masks1_folder:
    :param masks2_folder:
    :param results_path:
    :param show_results:
    :return:
    '''


    # Check if we have the same number of patients in both folders (nnunet_preds_dataset_path and total_seg_bones_dataset_path)
    masks1_list = os.listdir(masks1_folder)
    masks2_list = os.listdir(masks2_folder)
    mask1_file_ending_list = [file for file in masks1_list if file.endswith(mask1_file_ending)]
    mask2_file_ending_list = [file for file in masks2_list if file.endswith(mask2_file_ending)]


    print(mask1_file_ending_list)
    print(mask2_file_ending_list)

    # Check if the patients are the same in both folders (nnunet_preds_dataset_path and total_seg_bones_dataset_path)
    mask1_file_ending_list.sort()
    mask2_file_ending_list.sort()
    # for i in range(len(mask1_file_ending_list)):
    #     if mask1_file_ending_list[i].split('.')[0] not in mask2_file_ending_list[i].split('.')[0]:
    #         print(f'ERROR: Patients from the masks1:{masks1_folder} are not in masks2: {masks2_folder}')
    #         return



    # Subtract the bones to the predictions and groundtruth of the PTV_tot
    for i, patient in enumerate(mask1_file_ending_list):
        print(f'Patient {i}: {patient}')
        mask1_patient_path = os.path.join(masks1_folder, patient)
        patient_name = patient.split('.')[0]
        mask2_patient_path = os.path.join(masks2_folder, patient_name + '_0000_multilabel.nii.gz')

        if mask1_file_ending == '.nii.gz':
            mask1_patient_nifti = nib.load(mask1_patient_path)

            # Subtract the bones to the predictions
            mask_subtracted = subtract_masks_nifti(mask1_patient_path, mask2_patient_path)

            # Save the resulting subtract mask in the results folder in the .nii.gz format
            mask_subtracted_nifti = nib.Nifti1Image(mask_subtracted, affine=mask1_patient_nifti.affine)
            mask_subtracted_name = f'{patient.split(".")[0]}.nii.gz'
            mask_subtracted_path = os.path.join(results_path, mask_subtracted_name)
            nib.save(mask_subtracted_nifti, mask_subtracted_path)
        elif mask1_file_ending == '.npz':

            # Replace mask2_patient_path ending with .nii.gz
            mask2_patient_path = mask2_patient_path.replace('.npz', '.nii.gz')
            # Subtract the bones to the predictions
            mask1_patient_pred_subtracted, mask1_patient_target_subtracted = subtract_masks_npz_nifti(mask1_patient_path, mask2_patient_path)

            # Save the resulting subtract mask in the results folder in the .npz format
            np.savez_compressed(os.path.join(results_path, patient.split('.')[0]), pred=mask1_patient_pred_subtracted, target=mask1_patient_target_subtracted)



def subtract_masks_nifti(mask_path, mask2_path):
    '''
    This function will subtract the mask2 from the mask1 and return the result as a numpy array
    :param mask_path:
    :param mask2_path:
    :return:
    '''
    mask = nib.load(mask_path)
    mask = mask.get_fdata()

    mask2 = nib.load(mask2_path)
    mask2 = mask2.get_fdata()

    mask[mask2 > 0] = 0

    return mask


def subtract_masks_npz_nifti(mask1_path, mask2_path):
    '''
    This function will subtract the mask2 from the mask1 and return the result as a numpy array
    :param mask_path:
    :param mask2_path:
    :return:
    '''
    mask1_pred = np.load(mask1_path)['pred']
    mask1_target = np.load(mask1_path)['target']
    mask1_pred = mask1_pred.transpose(1,2,0)
    mask1_target = mask1_target.transpose(1,2,0)

    mask2 = nib.load(mask2_path)
    mask2 = mask2.get_fdata()
    mask2 = mask2.transpose(1, 0, 2)

    mask1_pred[mask2 > 0] = 0
    mask1_target[mask2 > 0] = 0

    return mask1_pred, mask1_target


def rename_bones_masks_to_patient_names(bone_masks_path, patient_dictionary_path, results_path):
    # Get the list of bone masks that end with .nii.gz in the bone_masks_path
    bone_masks_list = os.listdir(bone_masks_path)
    bone_masks_list = [file for file in bone_masks_list if file.endswith('multilabel.nii.gz')]
    bone_masks_list.sort()

    # Load the patient dictionary csv that contains the keys for the patient names
    name_mapping = {}
    with open(patient_dictionary_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            old_name = row[0]
            new_name = row[1]
            name_mapping[new_name] = old_name

    # Rename the bone masks to the patient names
    for bone_mask in bone_masks_list:
        bone_mask_name = bone_mask.split('.')[0]
        bone_mask_name = bone_mask_name.split('_')[0] + '_' + bone_mask_name.split('_')[1]
        new_bone_mask_name = f'{name_mapping[bone_mask_name]}.nii.gz'
        shutil.copy(os.path.join(bone_masks_path, bone_mask), os.path.join(results_path, new_bone_mask_name))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Subtract the masks of the second argument to the masks of the first '
                                                 'argument. The masks are expected to be in .nii.gz format.')
    parser.add_argument('--preds_path', type=str, help='Path to the folder containing the masks to subtract from')
    parser.add_argument('--bones_preds_path', type=str, help='Path to the folder containing the masks to subtract')
    parser.add_argument('--results_path', type=str, help='Path to the folder where the results will be saved')
    args = parser.parse_args()

    subtract_masks_folder(args.preds_path, args.bones_preds_path, args.results_path, mask1_file_ending='.npz', mask2_file_ending='.nii.gz')

    # bone_masks_path = '/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset001_AUTOMI100/total_segmentator_bone_segmentations/'
    # patient_dictionary_path = '/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset001_AUTOMI100/patient_dictionary.csv'
    # results_path = '/home/aiart/results/AUTOMI_PTV_tot_BCE_Loss_May_2023/totalseg_bone_segmentations'

    #rename_bones_masks_to_patient_names(bone_masks_path, patient_dictionary_path, results_path)
