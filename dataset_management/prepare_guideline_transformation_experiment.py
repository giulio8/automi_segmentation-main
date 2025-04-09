'''
This script  is used to prepare the dataset for the transformation of the old guidelines of CTV delineation to the new guidelines.
'''

import os
from renameRTStruct_structures import automi_missing_structures

def check_two_datasets_overlap(dataset1, dataset2):
    '''
    Check if the patients names in the two datasets overlap. Dataset2 patients names might be a substring of the Dataset1 patients names.'''
    dataset1_patients = [patient for patient in os.listdir(dataset1)]
    dataset2_patients = [patient for patient in os.listdir(dataset2)]
    overlapping_patients = []
    for patient1 in dataset1_patients:
        for patient2 in dataset2_patients:
            if patient2 in patient1:
                overlapping_patients.append(patient2)
    
    if len(overlapping_patients) == 0:
        return None
    return overlapping_patients


if __name__ == '__main__':
    # Define the paths to the datasets
    ptv_dataset = "/home/aiart/AUTOMI/imgs/"
    ctv_45_dataset = "/home/aiart/AUTOMI/AUTOMI_40_patients/original_dataset/"

    # Check if the datasets overlap
    overlapping_patients = check_two_datasets_overlap(ptv_dataset, ctv_45_dataset)
    print(f'Overlapping patients: {overlapping_patients}')

    # Check if the patients in the original dataset have the structure that is missing in the new dataset
    structure = 'PTV_tot'
    missing_structures = automi_missing_structures(ctv_45_dataset, structure, RTStruct_name='RS1')
    print(f'The patients that do not have the structure {structure} are: {missing_structures}')
    print(f'Number of patients in the dataset: {len(os.listdir(ctv_45_dataset))}  Number of patients missing the structure: {len(missing_structures)}')

    # List the patient names in the ptv dataset
    ptv_patients = [patient for patient in os.listdir(ptv_dataset)]
    print(f'Patients in the PTV dataset: {ptv_patients}')

    # List the patient names in the ctv_45 dataset
    ctv_45_patients = [patient for patient in os.listdir(ctv_45_dataset)]
    print(f'Patients in the CTV_45 dataset: {ctv_45_patients}')


# Conclusions: There are no overlapping patients between the two datasets. And there are 36 out of the 45 patients that do not have the PTV_tot structure.