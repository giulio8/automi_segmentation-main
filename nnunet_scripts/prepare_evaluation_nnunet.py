'''This script is used to prepare the evaluation folders for the nnUnet framework models'''

import os, sys
import shutil
import json
import nibabel as nib
import numpy as np
import argparse


def copy_test_patients(dataset_path, test_split_json_path, original_pred_folder, results_path):
    '''
    This function will copy the test patients to the results path.
    :param dataset_path: path to the nnUnet dataset
    :param test_split_json_path: .json file path with an entry for each fold and a list of patients in "test" that correspond to
    the patients in the test set for that fold
    :param results_path: path to the results folder
    :return:
    '''

    # Get the test patients
    test_patients = []
    with open(test_split_json_path, 'r') as f:
        test_split = json.load(f)
        for fold in test_split:
            print(fold['test'])
            test_patients.append(fold['test'])


    for i, test_fold in enumerate(test_patients):
        # print the number of patients in this test_fold
        print(f'Number of patients in fold {i}: {len(test_fold)}')
        print(f'List of patients in fold {i}: {test_fold}')

        # Create the test folder

        test_folder = os.path.join(results_path, f'fold_{i}')
        if not os.path.isdir(test_folder):
            os.makedirs(test_folder)

        test_img_folder = os.path.join(test_folder, 'test_img')
        if not os.path.isdir(test_img_folder):
            os.makedirs(test_img_folder)

        test_gt_folder = os.path.join(test_folder, 'test_gt')
        if not os.path.isdir(test_gt_folder):
            os.makedirs(test_gt_folder)

        test_pred_folder = os.path.join(test_folder, 'test_pred')
        if not os.path.isdir(test_pred_folder):
            os.makedirs(test_pred_folder)


        # Copy the test patients to the test folder
        for patient in test_fold:
            patient_img_name = patient + '_0000.nii.gz'
            patient_gt_name = patient + '.nii.gz'
            patient_pred_name = patient_gt_name

            patient_path = os.path.join(dataset_path, 'imagesTr', patient_img_name)
            shutil.copy(patient_path, test_img_folder)

            patient_gt_path = os.path.join(dataset_path, 'labelsTr', patient_gt_name)
            shutil.copy(patient_gt_path, test_gt_folder)

            patient_pred_path = os.path.join(original_pred_folder, patient_pred_name)
            shutil.copy(patient_pred_path, test_pred_folder)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare the evaluation folders for the nnUnet framework models')
    parser.add_argument('--dataset_path', type=str, required=True,
                      help='Path to the nnUnet dataset with the raw images. (Example: Dataset001_AUTOMI100)')
    parser.add_argument('--test_split_json_path', type=str, required=True,
                      help='Path to the .json file with the test split. It should be in the pre_processed folder')
    parser.add_argument('--original_pred_folder', type=str, required=True,
                        help='Path to the folder with the original predictions. (Example: Dataset001_AUTOMI100/3d_fullres/cross_val...')
    parser.add_argument('--results_path', type=str, required=True,
                      help='Path to the results folder. Should be the Dataset001_AUTOMI100/3d_fullres folder or other configuration.')
    args = parser.parse_args()

    copy_test_patients(args.dataset_path, args.test_split_json_path, args.original_pred_folder, args.results_path)