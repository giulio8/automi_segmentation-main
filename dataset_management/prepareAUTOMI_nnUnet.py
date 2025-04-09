'''This script is used to convert the AUTOMI dataset into the nnUnet format.'''

import os, sys
import shutil

from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils_data
import numpy as np
from PIL import Image
from dcmrtstruct2nii import dcmrtstruct2nii
import csv
import json
import argparse
import nibabel as nib


def convert_automi_to_nnUnet_png(images_path, new_images_path, masks_path, new_masks_path, structure):
    '''
    This function renames the images of the AUTOMI dataset (.png or .dcm) to the nnUnet format.
    The details of this format can be checked in the nnUnet documentation.
    :return:
    '''
    new_old_image_names = {}
    new_old_mask_names = {}

    # This number represents the modality of the image (e.g. CT, PET, etc.)
    MODALITY_NUM = '0000'
    identifier = 0

    for patient_name in os.listdir(images_path):
        patient_path = os.path.join(images_path, patient_name)
        patient_mask_path = os.path.join(masks_path, patient_name, f'{structure}.npz')
        print(f'Patient {patient_name}')
        if os.path.isfile(patient_mask_path):

            image_patient_list = os.listdir(patient_path)
            image_patient_list.sort()
            for i, image_name in enumerate(image_patient_list):
                image_path = os.path.join(patient_path, image_name)
                new_image_name = f'AUTOMI_{str(identifier).zfill(5)}_{MODALITY_NUM}.png'
                new_image_path = os.path.join(new_images_path, new_image_name)
                sliceNumber, pngSlice = utils_data.convertCTDicomToPNG(image_path)
                pngSlice.save(new_image_path)
                new_old_image_names[new_image_name] = image_name


                # Loads the mask from the 3d numpy array stored in the .npz file and saves it as a .png with the new name
                arr = np.load(patient_mask_path)
                masks3d = arr['arr_0']

                mask = masks3d[:, :, i]

                new_mask_name = f'AUTOMI_{str(identifier).zfill(5)}.png'
                new_mask_path = os.path.join(new_masks_path, new_mask_name)
                Image.fromarray(mask).save(new_mask_path)
                new_old_mask_names[new_mask_name] = f'{patient_name}_{structure}_{i}.png'

                identifier += 1

        else:
            print(f'No mask found for patient {patient_name} and structure {structure}.')

    # Saves the dictionary with the old and new names of the images as a .csv file
    with open(os.path.join(new_images_path, 'new_old_image_names.csv'), 'w') as f:
        for key in new_old_image_names.keys():
            f.write("%s,%s\n" % (key, new_old_image_names[key]))

    # Saves the dictionary with the old and new names of the masks as a .csv file
    with open(os.path.join(new_masks_path, 'new_old_mask_names.csv'), 'w') as f:
        for key in new_old_mask_names.keys():
            f.write("%s,%s\n" % (key, new_old_mask_names[key]))


def convert_automi_to_nnUnet_nii(original_dataset_path, new_dataset_path, structures, subString):
    ''' Converts all patients in the dataset to .nii.gz files. Including the entire CT scan and the mask of the structure
    using the library dcmrtstruct2nii.
    '''

    for patient in os.listdir(original_dataset_path):
        print(f'Patient {patient}')
        patient_path = os.path.join(original_dataset_path, patient)
        rt_struct_filename = utils_data.getPatientRTStructFileName(patient_path, subString)
        rt_struct_path = os.path.join(patient_path, rt_struct_filename)
        output_path = os.path.join(new_dataset_path, patient)

        #If there is already a image.nii.gz and a mask_*.nii.gz file, it skips the patient
        if os.path.isfile(os.path.join(output_path, 'image.nii.gz')) and os.path.isfile(os.path.join(output_path, f'mask_{structures[0]}.nii.gz')):
            print(f'Patient {patient} already converted.')
            continue
        if structures is None:
            dcmrtstruct2nii(rt_struct_path, patient_path, output_path)
        else:
            dcmrtstruct2nii(rt_struct_path, patient_path, output_path, structures=structures, mask_background_value=0, mask_foreground_value=1)


def rename_automi_nnUnet(nifti_dataset_path, nnUnet_dataset_path, structure):
    '''
    This function renames the images created by the convert_automi_to_nnUnet_nii function to the nnUnet naming format.
    :param nifti_dataset_path:
    :param nnUnet_dataset_path:
    :param structure:
    :return:
    '''
    patient_dic = {}

    # This number represents the modality of the image (e.g. CT, PET, etc.)
    MODALITY_NUM = '0000'
    identifier = 0
    patient_list = os.listdir(nifti_dataset_path)
    patient_list.sort()

    for patient in patient_list:
        patient_path = os.path.join(nifti_dataset_path, patient)

        ct_volume_path = os.path.join(patient_path, 'image.nii.gz')
        mask_volume_path = os.path.join(patient_path, f'mask_{structure}.nii.gz')

        new_ct_volume_path = os.path.join(nnUnet_dataset_path, 'imagesTr', f'AUTOMI_{str(identifier).zfill(5)}_{MODALITY_NUM}.nii.gz')
        new_mask_volume_path = os.path.join(nnUnet_dataset_path,'labelsTr', f'AUTOMI_{str(identifier).zfill(5)}.nii.gz')

        shutil.copy(ct_volume_path, new_ct_volume_path)
        shutil.copy(mask_volume_path, new_mask_volume_path)

        patient_dic[patient] = f'AUTOMI_{str(identifier).zfill(5)}'
        identifier += 1

    # Saves the dictionary with the old and new names of the images as a .csv file
    with open(os.path.join(nnUnet_dataset_path, 'patient_dictionary.csv'), 'w') as f:
        for key in patient_dic.keys():
            f.write("%s,%s\n" % (key, patient_dic[key]))

def rename_automi_nnUnet_multiple_structure_input(nifti_dataset_path, nnUnet_dataset_path, structures):
    '''
    This function renames the images created by the convert_automi_to_nnUnet_nii function to the nnUnet naming format.
    It is a special case of the function above and it is used when the goal is to insert multiple structures in the same
    nnUnet dataset as input.
    :param nifti_dataset_path:
    :param nnUnet_dataset_path:
    :param structures:
    :return:
    '''
    patient_dic = {}

    identifier = 0
    patient_list = os.listdir(nifti_dataset_path)
    patient_list.sort()
    for patient in patient_list:
        modality_num = 0
        patient_path = os.path.join(nifti_dataset_path, patient)
        ct_volume_path = os.path.join(patient_path, 'image.nii.gz')
        new_ct_volume_path = os.path.join(nnUnet_dataset_path,
                                          'imagesTr',
                                          f'AUTOMI_{str(identifier).zfill(5)}_{str(modality_num).zfill(4)}.nii.gz')
        shutil.copy(ct_volume_path, new_ct_volume_path)

        for structure in structures:
            modality_num += 1
            mask_volume_path = os.path.join(patient_path, f'{structure}.nii.gz')
            new_mask_volume_path = os.path.join(nnUnet_dataset_path,
                                                    'imagesTr',
                                                    f'AUTOMI_{str(identifier).zfill(5)}_{str(modality_num).zfill(4)}.nii.gz')
            if os.path.isfile(mask_volume_path):
                shutil.copy(mask_volume_path, new_mask_volume_path)
            else:
                #Creates a mask with the same shape as the image with all pixels set to 0
                print(f'Mask {mask_volume_path} not found. Creating a mask with all pixels set to 0.')
                image = nib.load(ct_volume_path)
                mask = np.zeros(image.get_fdata().shape, dtype=np.uint8)
                mask = nib.Nifti1Image(mask, image.affine, image.header)
                nib.save(mask, new_mask_volume_path)


        patient_dic[patient] = f'AUTOMI_{str(identifier).zfill(5)}'
        identifier += 1

    # Saves the dictionary with the old and new names of the images as a .csv file
    with open(os.path.join(nnUnet_dataset_path, 'patient_dictionary.csv'), 'w') as f:
        for key in patient_dic.keys():
            f.write("%s,%s\n" % (key, patient_dic[key]))

def rename_total_seg_nnUnet(nifti_dataset_path, final_mask_name, nnUnet_dataset_path, input_identifier='0001'):
    '''
    This function takes the nifti dataset created by the TotalSegmentator and renames the final mask images to the
    nnUnet format. It takes into account the input identifier to create the new names of the images.
    :param nifti_dataset_path:
    :param nnUnet_dataset_path:
    :param input_identifier:
    :return:
    '''

    for patient_folder in os.listdir(nifti_dataset_path):
        patient_path = os.path.join(nifti_dataset_path, patient_folder)
        mask_path = os.path.join(patient_path, f'{final_mask_name}.nii.gz')
        patient_number = patient_folder.split('_')[-2]
        new_mask_path = os.path.join(nnUnet_dataset_path, 'imagesTr', f'AUTOMI_{patient_number}_{input_identifier}.nii.gz')
        shutil.copy(mask_path, new_mask_path)


def join_masks_in_one(mask_path, new_mask_path, structures, structures_values, name_composite_mask='composite_mask'):
    '''
    This function joins the masks of the structures passed as parameter in a single mask in .nii.gz format, each
    structure will be assigned a different value from 0 to 255.
    :param mask_path:
    :param new_mask_path:
    :param structures:
    :return:
    '''

    # Get the list of masks
    mask_path_list = []

    for structure in structures:
        #mask_structure_path = os.path.join(mask_path, f'mask_{structure}.nii.gz')
        mask_structure_path = os.path.join(mask_path, f'{structure}.nii.gz')
        if os.path.isfile(mask_structure_path):
            mask_path_list.append(mask_structure_path)
        else:
            print(f'Mask {mask_structure_path} not found.')


    # Read each mask and add it to the final mask
    final_mask = None
    for i, mask in enumerate(mask_path_list):
        print(f'Mask {mask}')
        mask = nib.load(mask)
        mask_data = mask.get_fdata().astype(np.uint8)

        print(f'Mask data shape: {mask_data.shape}')
        print(f'Mask data type: {mask_data.dtype}')
        print(f'Mask max and min: {mask_data.max(), mask_data.min()}')

        if final_mask is None:
            final_mask = np.zeros(mask_data.shape, dtype=np.uint8)


        final_mask[mask_data == 1] = structures_values[i]

    # Save the final mask
    final_mask = nib.Nifti1Image(final_mask, mask.affine, mask.header)
    nib.save(final_mask, os.path.join(new_mask_path, f'mask_{name_composite_mask}.nii.gz'))


def join_masks_in_one_dataset(nifti_dataset, structures, structures_values, name_composite_mask='composite_mask'):
    '''
    This function joins the masks of the structures passed as parameter in a single mask in .nii.gz format, each
    structure will be assigned a different value from 0 to 255.
    '''

    patient_list = os.listdir(nifti_dataset)
    patient_list.sort()

    for patient in patient_list:
        patient_path = os.path.join(nifti_dataset, patient)
        join_masks_in_one(patient_path, patient_path, structures, structures_values, name_composite_mask=name_composite_mask)


def use_composite_mask_to_crop_images(nifti_dataset, composite_mask_name, cropping_value=0,
                                      cropped_image_name='experiment3_cropping_value'):
    '''
    This function uses the composite mask to crop the images in the dataset. The region of the composite mask is the region
    that is maintained in the cropped images. The region outside the composite mask is set to the value passed as parameter
    (cropping_value).
    :param nifti_dataset: The folder to the dataset with the .nii.gz images of the dataset
    :param composite_mask_name: The name of the composite mask in the dataset that will be used to crop the images.
    :param new_nifti_dataset: The name of the folder where the cropped images will be saved.
    :param cropping_value: The value of the pixels outside the composite mask. For experiment 3 it can be either 0 or 255.
    :return:
    '''

    patient_list = os.listdir(nifti_dataset)
    patient_list.sort()

    for patient in patient_list:
        patient_path = os.path.join(nifti_dataset, patient)
        image_path = os.path.join(patient_path, 'image.nii.gz')
        mask_path = os.path.join(patient_path, f'mask_{composite_mask_name}.nii.gz')

        image = nib.load(image_path)
        mask = nib.load(mask_path)

        image_data = image.get_fdata()
        mask_data = mask.get_fdata()


        # Set the pixels outside the mask to the cropping value
        image_data[mask_data == 0] = cropping_value

        image = nib.Nifti1Image(image_data, image.affine, image.header)

        new_image_path = os.path.join(patient_path, f'{cropped_image_name}.nii.gz')
        nib.save(image, new_image_path)


def patient_split_to_nnUnet_format(patient_fold_split_path, patient_nnunet_dict_path):
    ''' Converts the patient split to the format used by nnUnet. Should be run after the patient data has been converted to the nnUnet format,
    and after the naming dictionary has been created.
    '''

    patients = {}
    with open(patient_fold_split_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row['patientID']
            fold = int(row['fold'])
            patients[name] = fold

    # Read the mapping from old names to new names from the second CSV file
    name_mapping = {}
    with open(patient_nnunet_dict_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            old_name = row[0]
            new_name = row[1]
            name_mapping[old_name] = new_name

    json_data = []
    json_test_data = []
    train_fold_list = [[1,2,3],[2,3,4], [3,4,5], [4,5,1], [5,1,2]]
    val_fold_list = [[4], [5], [1], [2], [3]]
    test_fold_list = [[5], [1], [2], [3], [4]]
    for i in range(1, 6):
        train_fold = [(name_mapping.get(name), fold) for name, fold in patients.items() if
                      fold in train_fold_list[i-1]]
        val_fold = [(name_mapping.get(name), fold) for name, fold in patients.items() if fold in val_fold_list[i-1]]
        test_fold = [(name_mapping.get(name), fold) for name, fold in patients.items() if fold in test_fold_list[i-1]]

        train_data = [new_name for new_name, _ in train_fold if new_name is not None]
        val_data = [new_name for new_name, _ in val_fold if new_name is not None]
        test_data = [new_name for new_name, _ in test_fold if new_name is not None]

        json_entry = {
            'train': train_data,
            'val': val_data
        }
        json_data.append(json_entry)

        json_test_entry = {
            'test': test_data
        }
        json_test_data.append(json_test_entry)

    # Save the JSON data to a file
    with open('splits_final.json', 'w') as jsonfile:
        json.dump(json_data, jsonfile, indent=4)

    # Save the JSON test data to a file
    with open('test_split.json', 'w') as jsonfile:
        json.dump(json_test_data, jsonfile, indent=4)


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, required=True, help='Dataset to convert to nnUnet format')
    # parser.add_argument('--structures', type=str, nargs='+', default=['CTV_LNF_NGL'], help='Structures to convert to nnUnet format')
    # parser.add_argument('--output_path', type=str, required=True, help='Path to save the converted dataset')
    # args = parser.parse_args()
    #
    # dataset = args.dataset
    # new_dataset_path = args.output_path
    # structures = args.structures

    # convert_automi_to_nnUnet_nii(dataset, new_dataset_path, structures, subString='RS1')

    # dataset_path = "/mnt/storage/ricardo/nnUnetAUTOMI/dataset/nnUnet_raw/nifti_ct_multiple_inputs/"
    # structures = ['Heart', 'Liver', 'Spleen', 'Stomach', 'Eye_L', 'Eye_R', 'Kidney_L', 'Kidney_R', 'Femur_Head_L', 'Femur_Head_R']
    # num_structures = len(structures)
    # structure_value_increment = 255 // num_structures
    # structures_values = [structure_value_increment * i for i in range(1, num_structures + 1)]
    # structures_values = [255] */ num_structures
    # structures_values = [75, 100, 175, 125, 100, 100, 200, 200, 255, 255]
    # join_masks_in_one_dataset(dataset_path, structures, structures_values, name_composite_mask='add_input_9')


    # EXPERIMENT CTV_LNF_NGL ADDING INPUTS - 1
    # dataset_path = "/mnt/storage/ricardo/nnUnetAUTOMI/dataset/nnUnet_raw/nifti_ct_multiple_inputs/"
    # structures = ['Heart', 'Liver', 'Spleen', 'Stomach', 'Eye_L', 'Eye_R', 'Kidney_L', 'Kidney_R', 'Femur_Head_L', 'Femur_Head_R']
    # structures_values = [75, 100, 175, 125, 100, 100, 200, 200, 255, 255]
    # name_final_mask = 'mask_add_input_9'
    # join_masks_in_one_dataset(dataset_path, structures, structures_values, name_composite_mask=name_final_mask)
    # new_dataset_path = "/mnt/storage/ricardo/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset009_AUTOMI_CTV_1_multiple_inputs/"
    # rename_automi_nnUnet_multiple_structure_input(dataset_path, new_dataset_path, [name_final_mask])

    # EXPERIMENT CTV_LNF_NGL ADDING INPUTS - 2 - TotalSegmentator structures (workstation)
    #dataset_path = "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/total_segmentator_structures/"
    # structures = ['humerus_left', 'humerus_right', 'scapula_left', 'scapula_right', 'clavicula_left', 'clavicula_right',
    #               'femur_left', 'femur_right', 'hip_left', 'hip_right', 'sacrum', 'spleen', 'liver', 'stomach', 'urinary_bladder',
    #               'pancreas', 'kidney_right', 'kidney_left', 'iliopsoas_left', 'iliopsoas_right']
    # structures_values = [255,255,255,255,255,255,255,255,255,255,255,225,200, 175, 150, 125, 100, 75, 75, 50, 50]
    #name_final_mask = 'mask_mask_add_input_20_total_segmentator'
    #join_masks_in_one_dataset(dataset_path, structures, structures_values, name_composite_mask=name_final_mask)
    #new_dataset_path = "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset010_AUTOMI_CTV_2_multiple_inputs/"
    #rename_total_seg_nnUnet(dataset_path, name_final_mask, new_dataset_path, input_identifier='0001')


    # use_composite_mask_to_crop_images(dataset, 'intensity_255', cropping_value=0,
    #                                   cropped_image_name='experiment3_crop_eq_0')
    # use_composite_mask_to_crop_images(dataset, 'intensity_255', cropping_value=255,
    #                                   cropped_image_name='experiment3_crop_eq_255')

    # Splits  the patients according to the fold distribution in the previous experiments and creates the JSON file
    # with the split
    # patient_split_path = '/home/ricardo/Desktop/automi_segmentation/patientDist_CTVLNF_NEWGL.csv'
    # patient_split_path = '/home/ricardo/Desktop/automi_segmentation/patientDist.csv'
    # patient_dict_path = '/mnt/storage/ricardo/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset004_AUTOMI_PTV_heart/patient_dictionary.csv'
    # patient_split_to_nnUnet_format(patient_split_path, patient_dict_path)


