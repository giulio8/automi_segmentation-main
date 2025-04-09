'''
This script is used to predict several bone structures using the TotalSegmentator models.
'''

import os
import subprocess

input_directory = "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/imagesTr/"
output_directory = "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/total_segmentator_structures/"
#roi_subset = "vertebrae_L5 vertebrae_L4 vertebrae_L3 vertebrae_L2 vertebrae_L1 vertebrae_T12 vertebrae_T11 vertebrae_T10 vertebrae_T9 vertebrae_T8 vertebrae_T7 vertebrae_T6 vertebrae_T5 vertebrae_T4 vertebrae_T3 vertebrae_T2 vertebrae_T1 vertebrae_C7 vertebrae_C6 vertebrae_C5 vertebrae_C4 vertebrae_C3 vertebrae_C2 vertebrae_C1 rib_left_1 rib_left_2 rib_left_3 rib_left_4 rib_left_5 rib_left_6 rib_left_7 rib_left_8 rib_left_9 rib_left_10 rib_left_11 rib_left_12 rib_right_1 rib_right_2 rib_right_3 rib_right_4 rib_right_5 rib_right_6 rib_right_7 rib_right_8 rib_right_9 rib_right_10 rib_right_11 rib_right_12 humerus_left humerus_right scapula_left scapula_right clavicula_left clavicula_right femur_left femur_right hip_left hip_right sacrum "
#roi_subset = "humerus_left humerus_right scapula_left scapula_right clavicula_left clavicula_right femur_left femur_right hip_left hip_right sacrum spleen liver stomach urinary_bladder pancreas kidney_right kidney_left iliopsoas_left iliopsoas_right"
roi_subset = "vertebrae_L4 vertebrae_T1"
# Check if the input directory exists
if not os.path.exists(input_directory):
    print(f"Input directory does not exist: {input_directory}")
    exit(1)

# Check if the output directory exists, if not, create it
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Loop through the files in the input directory
for patient_file in os.listdir(input_directory):
    patient_file_path = os.path.join(input_directory, patient_file)

    # Check if the file is a regular file
    if os.path.isfile(patient_file_path):

        # Extract the filename without extension to name a directory
        patient_directory_name = patient_file.split(".")[-3]
        patient_output_directory = os.path.join(output_directory, patient_directory_name)

        # Check if the patient output directory exits, if not, create it
        if not os.path.exists(patient_output_directory):
            os.makedirs(patient_output_directory)

        for roi in roi_subset.split(" "):
            # Check if the roi .nii.gz file exists
            roi_file_path = os.path.join(patient_output_directory, f"{roi}.nii.gz")
            if os.path.exists(roi_file_path):
                print(f"ROI file already exists: {roi_file_path}")
            else:
                # Build the command
                #command = f"TotalSegmentator -i '{patient_file_path}' -o '{patient_output_directory}' --roi_subset '{roi}' --ta total -ml"
                command = f"TotalSegmentator -i '{patient_file_path}' -o '{patient_output_directory}' --roi_subset '{roi}'"
                print("Running:", command)
                # Execute the command
                subprocess.run(command, shell=True, check=True)


