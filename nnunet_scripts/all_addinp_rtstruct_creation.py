'''
This script is used to create RTSTRUCT files for all the patients in the dataset.
The RTSTRUCT files are created from the nifti segmentation files. and will include the CTV_pred region
from all the additional input experiments.
'''

from rt_utils import RTStructBuilder
import nibabel as nib
import numpy as np
import pandas as pd
import os
import utils_data

# Path to the nifti segmentation files of all datasets

dataset_pred_path_list = ['/home/ricardo/Desktop/AddInputImages/Dataset_003_predictions/',
                          '/home/ricardo/Desktop/AddInputImages/Dataset_005_predictions/',
                          '/home/ricardo/Desktop/AddInputImages/Dataset_006_predictions/',
                          '/home/ricardo/Desktop/AddInputImages/Dataset_007_predictions/',
                          '/home/ricardo/Desktop/AddInputImages/Dataset_009_predictions/',
                          '/home/ricardo/Desktop/AddInputImages/Dataset_010_predictions/']

# Path to the original dataset
original_dataset_path = '/home/ricardo/Desktop/AddInputImages/original_dataset/'

# Path to the original DICOM FILES

# Find the patient dictionary csv that has the original_name and the nnunet_name as columns
patient_dict_path = '/mnt/storage/ricardo/AUTOMI/AUTOMI_40_patients/patient_dictionary.csv'
patient_dict = pd.read_csv(patient_dict_path)

for patient in os.listdir(dataset_pred_path_list[0]):
    patient_nnunet_name = patient.split('.')[0]
    patient_original_name = patient_dict[patient_dict['nnunet_name'] == patient_nnunet_name]['original_name'].values[0]
    print(f'Creating RTSTRUCT for patient: {patient_original_name} / {patient_nnunet_name}')
    patient_dicom_path = os.path.join(original_dataset_path, patient_original_name)
    # Find RTSTRUCT file
    if patient_original_name == '330f5b66ft' or patient_original_name == '330f5b66st':
        pass
    else:
        rt_struct_file = utils_data.getPatientRTStructFileName(patient_dicom_path, subString='RS')
        rt_struct_path = os.path.join(patient_dicom_path, rt_struct_file)
        rtstruct = RTStructBuilder.create_from(dicom_series_path=patient_dicom_path,
                                               rt_struct_path=rt_struct_path)
        roi_number_big = 300
        for dataset_pred_path in dataset_pred_path_list:
            # Read nifti segmentation prediction file
            nifti_pred = nib.load(os.path.join(dataset_pred_path, patient))
            nifti_pred = nifti_pred.get_fdata()
            nifti_pred = nifti_pred.transpose(1, 0, 2)
            # flip the nifti_pred to match the DICOM coordinate system
            nifti_pred = np.flip(nifti_pred, axis=0)
            nifti_pred = np.round(nifti_pred / np.max(nifti_pred)).astype("int8")
            nifti_pred = nifti_pred > 0

            rtstruct.add_roi(mask=nifti_pred, name=f'CTV_pred_{dataset_pred_path.split("/")[-2]}', roi_number=roi_number_big)
            roi_number_big += 1

        rtstruct.save(os.path.join(patient_dicom_path, f'{patient_original_name}.dcm'))





