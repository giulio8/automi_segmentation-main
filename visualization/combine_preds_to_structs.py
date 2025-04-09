# This script is used to combine several predictions that are in .npz and .nii.fz format into a single rtstruct that can be visualized after


import os, sys
import numpy as np
# appen the parent directory to the sys path so we can import the utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils_data
import nibabel
import utils_evaluation
from rt_utils import RTStructBuilder
import argparse

def add_preds_to_rt_struct(pred_folder_path, dataset_dicom_path, original_dicom_path, results_path):
    for patient_name in os.listdir(pred_folder_path):
        patient_pred_folder_path = os.path.join(pred_folder_path, patient_name)

        # Original Patient DICOM images path
        patient_dicom_path = os.path.join(dataset_dicom_path, patient_name)
        patient_original_dicom_path = os.path.join(original_dicom_path, patient_name)

        rt_struct_patient_filename = utils_data.getPatientRTStructFileName(patient_original_dicom_path, subString='RTSTRUCT')
        rt_struct_patient_path = os.path.join(patient_original_dicom_path, rt_struct_patient_filename)

        # Create RTStruct
        print('Creating RTStruct for patient: ', patient_name)
        rt_struct = RTStructBuilder.create_from(dicom_series_path=patient_dicom_path,
                                                rt_struct_path=rt_struct_patient_path)

        roi_number = 101
        for pred in os.listdir(patient_pred_folder_path):
            print('Adding ROI: ', pred)
            pred_path = os.path.join(patient_pred_folder_path, pred)
            if pred.endswith('noBones.npz'):
                pred_array = np.load(pred_path)
                pred_array = pred_array['pred']
                pred_array = utils_evaluation.post_process_prediction(pred_array)
            elif pred.endswith('.npz'):
                pred_array = np.load(pred_path)
                pred_array = pred_array['pred']
                pred_array = pred_array.transpose(1, 2, 0)
                pred_array = utils_evaluation.post_process_prediction(pred_array)
            elif pred.endswith('.nii.gz'):
                pred_array = nibabel.load(pred_path).get_fdata()
                pred_array = pred_array.transpose(1, 0, 2)
                pred_array = np.round(pred_array / np.max(pred_array)).astype("int8")
                pred_array = pred_array > 0
            else:
                print('Error: Prediction file is not .npz or .nii.gz')
                sys.exit(1)

            print(pred_array.shape)
            #Add ROI to RTStruct
            rt_struct.add_roi(mask=pred_array, name=pred, use_pin_hole=True, roi_number=roi_number)
            roi_number += 1

        rt_struct.save(os.path.join(results_path, patient_name + '.dcm'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_folder_path', type=str, required=True)
    parser.add_argument('--dataset_dicom_path', type=str, required=True)
    parser.add_argument('--original_dicom_path', type=str, required=True)
    parser.add_argument('--results_path', type=str, required=True)

    args = parser.parse_args()

    add_preds_to_rt_struct(args.pred_folder_path, args.dataset_dicom_path, args.original_dicom_path, args.results_path)

