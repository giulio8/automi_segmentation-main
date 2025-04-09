'''
This script will be able to rename the structures of the groundtruth inside the RTStruct files.
For example, it will detect the patients that not have a groundtruth named 'PTV_tot' and it print their names so they can be renamed.
'''

import os, sys
import pydicom
#include the path to the directory above
sys.path.append(".")
sys.path.append("..")
import utils_data



def find_patient_structure(rtstruct_path, structure):
    '''
    This function will return True if the patient possesses the specific structure and False if not.
    :param rtstruct_path:
    :param structure:
    :return:
    '''
    found_structure = False

    # Read the RTStruct file
    rtstruct = pydicom.dcmread(rtstruct_path)

    # Get the structures
    structures = rtstruct.StructureSetROISequence

    # Check if the structure is in the RTStruct file
    for struct in structures:
        if struct.ROIName == structure:
            found_structure = True
            return found_structure

    return found_structure


def automi_missing_structures(original_dataset_path, structure, RTStruct_name='RS1'):
    '''
    This function will return the patients that not have the structure that is passed as parameter.
    :param original_dataset_path:
    :param structure:
    :return:
    '''
    patients = []
    for patient in os.listdir(original_dataset_path):
        patient_path = os.path.join(original_dataset_path, patient)
        rtstruct_filename = utils_data.getPatientRTStructFileName(patient_path, RTStruct_name)
        rtstruct_path = os.path.join(patient_path, rtstruct_filename)

        if not find_patient_structure(rtstruct_path, structure):
            patients.append(patient)

    return patients


def rename_wrong_structures_rtstruct(original_dataset_path, wrongly_named_patients, wrong_labels, correct_label, RTStruct_name='RS1'):
    '''
    This function will rename the structures of the RTStruct files that are wrongly named and replace them with the correct ones.
    :param original_dataset_path: the path to the orignal dataset
    :param wrongly_named_patients: list of patients with wrong named structure
    :param wrong_labels: ordered list of the wrong named labels of that structure
    :param correct_label: name of the correct label
    :return:
    '''
    patient_number = 1
    for patient in wrongly_named_patients:
        patient_path = os.path.join(original_dataset_path, patient)
        rtstruct_filename = utils_data.getPatientRTStructFileName(patient_path, RTStruct_name)
        rtstruct_path = os.path.join(patient_path, rtstruct_filename)

        # Read the RTStruct file
        rtstruct = pydicom.dcmread(rtstruct_path)

        # Get the structures
        structures = rtstruct.StructureSetROISequence

        # Check if the structure is in the RTStruct file
        for struct in structures:
            if struct.ROIName in wrong_labels:
                old_label = struct.ROIName
                struct.ROIName = correct_label
                print(f'{patient_number} - Patient {patient}\'structure has been renamed from {old_label} to {correct_label}')

        patient_number += 1
        # Save the RTStruct file
        rtstruct.save_as(rtstruct_path)


if __name__ == '__main__':
    #original_dataset_path = datapaths.original_datasets['AUTOMI']
    original_dataset_path = '/mnt/storage/ricardo/AUTOMI/AUTOMI_40_patients/original_dataset'
    structure = 'Heart'

    patients = automi_missing_structures(original_dataset_path, structure, 'RS1')
    print(patients)
    wrong_labels = ['Heart_Inferior_Left_PA']
    rename_wrong_structures_rtstruct(original_dataset_path, patients, wrong_labels, structure, 'RS1')