''' This scripts uses the predictions and groundtruth .nii.gz files to create the error maps as a .nii.gz file.'''
import os

import numpy as np
import nibabel as nib
from utils_evaluation import calcDiceScore, calcHausdorffMAD
import pandas as pd


def error_map(prediction_nii_file_path, ground_truth_nii_file_path, threshold=0.5, include_metrics=False):

    prediction_nii = nib.load(prediction_nii_file_path)
    ground_truth_nii = nib.load(ground_truth_nii_file_path)

    # Get data from NIfTI files
    prediction_mask = prediction_nii.get_fdata()
    ground_truth_mask = ground_truth_nii.get_fdata()

    # Calculate pixel-wise differences
    pred_gt_error_map = np.abs(prediction_mask - ground_truth_mask)

    # Apply thresholding (optional)
    pred_gt_error_map[pred_gt_error_map < threshold] = 0
    pred_gt_error_map[pred_gt_error_map >= threshold] = 1

    # Create a new NIfTI image object for the error map
    error_map_nii = nib.Nifti1Image(pred_gt_error_map, prediction_nii.affine)

    dsc_slices_values = 0
    hd95_slices_values = 0

    if include_metrics:
        # Calculates DSC and HD95 metrics for each individual slice and stores it in a dictionary.
        dsc_slices_values = []
        hd95_slices_values = []
        for i in range(prediction_mask.shape[2]):
            pred_slice = prediction_mask[:, :, i]
            gt_slice = ground_truth_mask[:, :, i]
            print(f"Calculating metrics for slice {i} ou of {prediction_mask.shape[2]}")
            print(f"Prediction slice shape: {pred_slice.shape}")
            print(f"Max value in prediction slice: {np.max(pred_slice)}")

            # If the max of the prediction or gt slice is 0, it means that the slice is empty and we should skip it
            if np.max(pred_slice) == 0 or np.max(gt_slice) == 0:
                dsc_slices_values.append(0)
                hd95_slices_values.append(0)
            else:
                dsc_slices_values.append(calcDiceScore(pred_slice, gt_slice))
                hd, mad, hd95 = calcHausdorffMAD(pred_slice, gt_slice)
                hd95_slices_values.append(hd95)

        return error_map_nii, dsc_slices_values, hd95_slices_values

    return error_map_nii, dsc_slices_values, hd95_slices_values


def dataset_error_maps(prediction_dataset_path, ground_truth_dataset_path, output_dataset_path, threshold=0.5, include_metrics=True):

    for patient in os.listdir(prediction_dataset_path):
        print(f"Creating error map for patient {patient}")
        patient_prediction_file_path = os.path.join(prediction_dataset_path, patient)
        patient_ground_truth_file_path = os.path.join(ground_truth_dataset_path, patient)
        patient_error_map, dsc_slice_values, hd95_slice_values = error_map(patient_prediction_file_path,
                                                                           patient_ground_truth_file_path,
                                                                           threshold,
                                                                           include_metrics=include_metrics)
        patient_error_map_file_path = os.path.join(output_dataset_path, patient)
        nib.save(patient_error_map, patient_error_map_file_path)

        # Save metrics to a .csv file
        if include_metrics:
            metrics_file_path = os.path.join(output_dataset_path, f"{patient}_metrics.csv")
            metrics = pd.DataFrame({'DSC': dsc_slice_values, 'HD95': hd95_slice_values})
            metrics.to_csv(metrics_file_path, index=False)


if __name__ == "__main__":
    prediction_dataset_path = "/home/ricardo/Desktop/CTV_LN_1"
    ground_truth_dataset_path = "/mnt/storage/ricardo/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset005_AUTOMI_CTV_1_multiple_inputs/labelsTr"
    output_dataset_path = "/home/ricardo/Desktop/CTV_LN_1_error_maps"
    threshold = 0.5
    dataset_error_maps(prediction_dataset_path, ground_truth_dataset_path, output_dataset_path, threshold)

    print("Error maps created successfully!")

