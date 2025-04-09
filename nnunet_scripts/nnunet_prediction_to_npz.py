import os, sys
import nibabel as nib
import numpy as np
import argparse

def nnunet_predictions_to_npz(nifti_gt_path, nifti_pred_path, npz_path):
    '''
    This function transforms the nifti files from the predictions and groundtruths of the nnunet to .npz files
    The .npz file contains two arrays: pred and target, which correspond to the prediction of the nnunet
     and groundtruth of the dataset.
    :param nifti_gt_path: path of the folder to the groundtruth nifti files
    :param nifti_pred_path: path of the folder to the predictions nifti files
    :param npz_path: path of the folder where the .npz files will be saved
    :return: nothing
    '''

    if not os.path.isdir(npz_path):
        os.makedirs(npz_path)

    pred_list = os.listdir(nifti_pred_path)

    #Remove the files that don't have the .nii.gz extension
    pred_list = [file for file in pred_list if file.endswith('.nii.gz')]

    for pred in pred_list:
        patient_pred_path = os.path.join(nifti_pred_path, pred)
        #patient_name = pred.split('.')[0]
        #patient_gt_path = os.path.join(nifti_gt_path, patient_name + '_0000.nii.gz')
        patient_gt_path = os.path.join(nifti_gt_path, pred)
        patient_pred = nib.load(patient_pred_path).get_fdata()
        patient_gt = nib.load(patient_gt_path).get_fdata()
        np.savez_compressed(os.path.join(npz_path, pred.split('.')[0]), pred=patient_pred, target=patient_gt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transforms the nifti files from the predictions and groundtruths of the nnunet to .npz files')
    parser.add_argument('--nifti_gt_path', type=str, required=True, help='Path to the folder with the groundtruth nifti files')
    parser.add_argument('--nifti_pred_path', type=str, required=True, help='Path to the folder with the predictions nifti files')
    parser.add_argument('--npz_path', type=str, required=True, help='Path to the folder where the .npz files will be saved')
    args = parser.parse_args()

    nnunet_predictions_to_npz(args.nifti_gt_path, args.nifti_pred_path, args.npz_path)