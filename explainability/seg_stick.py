'''
This script is going to be used for the creation of segmentation sticks, that are sticks that represent each patient.
The stick will be divided vertically in anatomical regions, for example: head, neck, thorax, abdomen, pelvis, upper limbs, lower limbs.
Each region will be color coded according to the performance of a segmentation model in that region.
For example, if the model is good segmenting lymph nodes in the head, the head region will be colored in green, if the model is bad
 segmenting the head, the head region will be colored in red.
 To read the separation of the regions, the script will read a json file that contains the separation of the regions.
 Each patient will have the beginning and ending slice number of each region.
 The script will calculate the DSC and HD of each region and will color code the region according to the performance.
 The stick will be a stick figure image and the outline of the stick figure will be colored vertically according to
 its own region separation. The number of regions will be the same for all patients and for the stick image.
'''

import os
import subprocess
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import json
import sys
import csv
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_evaluation import calcDiceScore, calcHausdorffMAD
import seg_metrics.seg_metrics as sg
import pydicom


STICK_BASE_PNG = "/home/aiart/AUTOMI/automi_segmentation/explainability/seg_stick_v3.png"
#STICK_BASE_REGIONS = [[0, 120], [126, 166], [171, 252], [258, 311], [317, 363], [369, 505]] # Head, Neck, Thorax, Abdomen, Pelvis, Lower limbs
STICK_BASE_REGIONS = [[0, 166], [171, 252], [258, 311], [317, 505]]

def get_dice_color(dice, norm='none'):
    '''
    Colormap the dice coefficient to a RGB color
    :param dice:
    :return: RGB color
    '''
    cmap = plt.get_cmap("viridis")
    if norm == None:
        return cmap(dice)
    elif norm == 'agd':
        # The dice is between 0 and 0.10, so we need to get a value between 0 and 1
        return cmap(dice/0.15)
    elif norm == 'mdg':
        return cmap(dice/0.15)
    elif norm == 'qd':
        # The dice is between 0 and 0.15, so we need to get a value between 0 and 1
        return cmap(dice/0.15)


def get_hd_color(hd):
    '''
    Colormap the Hausdorff distance to a RGB color
    :param hd:
    :return: RGB color
    '''
    # Normalize the Hausdorff distance to the range [0, 1],maximum is around 20 and minimum is 0
    hd = hd / 20
    cmap = plt.get_cmap("viridis")
    return cmap(hd)

def calculate_regions_metric(regions, gt, pred, metric):
    region_metrics = []
    for region in regions:
        gt_region = gt[:,:,region[0]:region[1]]
        pred_region = pred[:,:,region[0]:region[1]]
        region_metrics.append(calc_region_dice_hd95(gt_region, pred_region, metric))
    return region_metrics

def calculate_regions_metric_3D(regions, gt, pred, spacing, metrics):
    region_metrics = []
    for region in regions:
        gt_region = gt[:,:,region[0]:region[1]]
        pred_region = pred[:,:,region[0]:region[1]]
        sg_metrics = sg.write_metrics(labels=[1], gdth_img=gt_region, pred_img=pred_region,
                                csv_file=None,  spacing=spacing, metrics=['dice'])
        region_metrics.append(sg_metrics[0]['dice'][0])
    return region_metrics


def binary_compare(predPath, spacing, metrics=None, target_normalized=False):
    if metrics is None:
        metrics = ['dice', 'hd', 'hd95']

    try:
        arrayNPZ = np.load(predPath)
        pred = arrayNPZ['pred']
        target = arrayNPZ['target']
        pred = np.round(pred / np.max(pred)).astype("int8")
        if target_normalized:
            target = np.round(target / np.max(target)).astype("int8")
        else:
            target = np.round(target / 255).astype("int8")
    except FileNotFoundError:
        print("File not existing")
        return None

    if target.shape == pred.shape:
        return sg.write_metrics(labels=[1], gdth_img=target, pred_img=pred,
                                csv_file=None,  spacing=spacing, metrics=metrics)
    else:
        print(f'Shape mismatch. pred={str(pred.shape)} and gt={str(target.shape)}')
        return None


def calc_region_dice_hd95(gt, pred, metric):
    dsc_slices_values = []
    hd95_slices_values = []
    for i in range(pred.shape[2]):
        pred_slice = pred[:, :, i]
        gt_slice = gt[:, :, i]

        # If the max of the prediction or gt slice is 0, it means that the slice is empty and we should skip it
        if np.max(pred_slice) == 0 or np.max(gt_slice) == 0:
            dsc_slices_values.append(0)
            hd95_slices_values.append(0)
        else:
            if metric == "dice":
                dsc_slices_values.append(calcDiceScore(pred_slice, gt_slice))
            elif metric == "hd95":
                hd, mad, hd95 = calcHausdorffMAD(pred_slice, gt_slice)
                hd95_slices_values.append(hd95)

    if metric == "dice":
        return np.mean(dsc_slices_values)
    elif metric == "hd95":
        return np.mean(hd95_slices_values)

def read_patient_regions_file(json_file):
    with open(json_file, "r") as f:
        patients = json.load(f)
    return patients

def create_seg_stick(patient_name, region_colors, stick_base_png, stick_base_regions, output_directory):
    # Load the stick base image
    stick_base = plt.imread(stick_base_png)

    # If the number of regions is different from the stick base regions, raise an error
    if len(region_colors) != len(stick_base_regions):
        raise ValueError("The number of regions is different from the stick base regions")

    # Color the regions of stick base image where the image is == 255 according to the patient region colors
    stick_base_colored = stick_base.copy()

    for i, region_color in enumerate(region_colors):
        stick_base_regions_begin = stick_base_regions[i][0]
        stick_base_regions_end = stick_base_regions[i][1]

        # If the pixel is that region and it's value is different from 0, color it with the region color based on the
        # performance of the model in that region.
        for x in range(stick_base.shape[0]):
            for y in range(stick_base.shape[1]):
                if stick_base_regions_begin <= x <= stick_base_regions_end:
                    if stick_base[x, y, 0] != 0:
                        stick_base_colored[x, y] = region_color
                else:
                    pass

    # Save the stick base colored image
    stick_base_colored_path = os.path.join(output_directory, f"{patient_name}_segstick.png")
    plt.imsave(stick_base_colored_path, stick_base_colored)


def get_top_bottom_index_binary_structure(binary_mask_path):
    '''
    Reads the binary mask from a .nii.gz file that was predicted by TotalSegmentator and returns the top and bottom index
    of the structure. The structure is the region where the binary mask is equal to 1.
    :param binary_mask_path:
    :return:
    '''

    binary_mask = nib.load(binary_mask_path).get_fdata()
    top_index = 0
    bottom_index = 0
    for i in range(binary_mask.shape[2]):
        if np.max(binary_mask[:,:,i]) == 1:
            top_index = i
            break
    for i in range(binary_mask.shape[2]-1, 0, -1):
        if np.max(binary_mask[:,:,i]) == 1:
            bottom_index = i
            break

    return top_index, bottom_index


def insert_region_indexes_patients_json(binary_structures_path, structure_list, output_json):
    '''
    Iterates the all the patient folders in the binary_structures_path, loads the binary masks that are specifiied
    in the structure list and extracts their beginning and ending slice indexes. The indexes are inserter for each patient
    and within each patient for each structure. The output is a json file that contains the indexes for each structure.
    :param binary_structures_path:
    :param structure_lists:
    :param output_json:
    :return:
    '''

    patients = {}
    for patient in os.listdir(binary_structures_path):
        patient_name = patient.split("_")
        patient_name = f"{patient_name[0]}_{patient_name[1]}"
        patient_path = os.path.join(binary_structures_path, patient)
        patient_structures = {}
        for structure in structure_list:
            structure_path = os.path.join(patient_path, f"{structure}.nii.gz")
            top_index, bottom_index = get_top_bottom_index_binary_structure(structure_path)
            patient_structures[structure] = [top_index, bottom_index]
        patients[patient_name] = patient_structures

    with open(output_json, "w") as f:
        json.dump(patients, f)


def get_patient_sex(patient):
    # Reads csv that maps the patient name both in nnUnet format and original format to patient's sex
    patient_map_csv = "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/patients_sex.csv"
    with open(patient_map_csv, "r") as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            if patient == row[1]:
                return row[2]
            else:
                print(f"Patient {patient} not found in the csv file")
    return None

def get_patient_original_name(patient):
    # Reads csv that maps the patient name both in nnUnet format and original format to patient's sex
    patient_map_csv = "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/patients_sex.csv"
    with open(patient_map_csv, "r") as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            if patient == row[1]:
                return row[0]
            else:
                print(f"Patient {patient} not found in the csv file")
    return None

def get_patient_sex_2(patient):
    # Reads csv that maps the patient name both in nnUnet format and original format to patient's sex and turns it into
    # a pandas dataframe
    patient_map_csv = "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/patients_sex.csv"
    patients_df = pd.read_csv(patient_map_csv)
    patient_found = patients_df[patients_df["patient_name_nnunet"] == patient]
    return patient_found['sex']


def create_region_metrics_and_segsticks_dataset(dataset_gt_path, dataset_prediction_path, dataset_output_path, patient_regions_file, metric):
    '''
    This function will calculate DSC and HD95 for each region of interest of each patient in the dataset and will create
    a stick figure image for each patient with the regions color coded according to the performance of the model in that region.
    It will also save the performance metrics, the sex and height of the patient in a .csv file, with the following columns.
    patient, sex, height, [metrics for the regions of interest]
    :param dataset_gt_path:
    :param dataset_prediction_path:
    :param dataset_output_path:
    :param patient_regions_file:
    :param metric:
    :return:
    '''
    patients = read_patient_regions_file(patient_regions_file)
    patient_rows = []
    for patient in patients:
        structures_indexes = patients[patient]
        vertebrae_T1_ind = structures_indexes["vertebrae_T1"]
        vertebrae_L4_ind = structures_indexes["vertebrae_L4"]
        stomach_ind = structures_indexes["stomach"]


        prediction_nii_file_path = os.path.join(dataset_prediction_path, f"{patient}.nii.gz")
        ground_truth_nii_file_path = os.path.join(dataset_gt_path, f"{patient}.nii.gz")

        # Check if the file exists and if not, skip the patient
        if not os.path.exists(prediction_nii_file_path) or not os.path.exists(ground_truth_nii_file_path):
            print(f"Prediction or ground truth file does not exist for patient {patient}")
            continue
        else:
            print(f"Calculating metrics for patient {patient}")
            prediction = nib.load(prediction_nii_file_path).get_fdata()
            ground_truth = nib.load(ground_truth_nii_file_path).get_fdata()

            # Gets patient's sex, height and regions of interest indexes
            patient_sex = get_patient_sex(patient)
            patient_height = prediction.shape[2]
            patient_original_name = get_patient_original_name(patient)
            original_dataset_path = '/home/aiart/AUTOMI/AUTOMI_40_patients/imgs/'
            patient_original_path = os.path.join(original_dataset_path, patient_original_name)
            listImages = os.listdir(patient_original_path)
            firstImagePath = os.path.join(patient_original_path, listImages[0])
            ds = pydicom.dcmread(firstImagePath)

            st = ds.SliceThickness
            ps = ds.PixelSpacing
            spacing = [st, ps[0], ps[1]]

            patient_regions = [[0, vertebrae_L4_ind[0]], [vertebrae_L4_ind[0]+1, stomach_ind[0]], [stomach_ind[0]+1,
                            vertebrae_T1_ind[0]], [vertebrae_T1_ind[0]+1, patient_height-1]]
            region_metrics = calculate_regions_metric_3D(patient_regions, ground_truth, prediction, spacing, metric)

            # Reverse the order of the region metrics to match the stick base regions (this way it becomes from top to bottom)
            region_metrics = region_metrics[::-1]
            if metric == "dice":
                region_colors = [get_dice_color(dice) for dice in region_metrics]
            elif metric == "hd95":
                region_colors = [get_hd_color(hd) for hd in region_metrics]

            create_seg_stick(patient, region_colors, STICK_BASE_PNG, STICK_BASE_REGIONS, dataset_output_path)
            patient_rows = [patient, patient_sex, patient_height] + region_metrics

            # Save the performance metrics
            with open(os.path.join(dataset_output_path, f"region_metric_{metric}.csv"), "a") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(patient_rows)

if __name__ == "__main__":
    # Read the patient regions file
    #patients = read_patient_regions_file("/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/patient_regions.json")

    # patient_name = 'AUTOMI_0001'
    # patient_regions = [[0, 50], [51, 70], [71, 132], [133, 161], [162, 171], [172, 225]]
    # prediction_nii_file_path = "AUTOMI_00001.nii.gz"
    # ground_truth_nii_file_path = "AUTOMI_00001_gt.nii.gz"
    # prediction_nii = nib.load(prediction_nii_file_path)
    # ground_truth_nii = nib.load(ground_truth_nii_file_path)
    #
    # # Get data from NIfTI files
    # prediction_mask = prediction_nii.get_fdata()
    # ground_truth_mask = ground_truth_nii.get_fdata()
    # create_seg_stick(patient_name, prediction_mask, ground_truth_mask, patient_regions, STICK_BASE_PNG, STICK_BASE_REGIONS, "/home/ricardo/Desktop/automi_segmentation/explainability")


    # EXPERIMENT 1 - CTV LNF NO MULTIPLE INPUTS
    # Creating the .json file with the indexes of the structures
    # insert_region_indexes_patients_json("/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/total_segmentator_structures",
    #                                     ["vertebrae_T1", "vertebrae_L4", "stomach"],
    #                                     "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/patient_regions.json")

    # Create the segmentation sticks for the dataset
    create_region_metrics_and_segsticks_dataset("/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/labelsTr",
                             "/home/aiart/nnUnetAUTOMI/dataset/nnUNet_results/Dataset003_AUTOMI_CTVLNF_NEWGL/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/",
                             "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/seg_sticks",
                             "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/patient_regions.json",
                             "dice")

    # Create individual segmentation stick with provided dice score values
    # patient_name = 'AUTOMI_average_male'
    # metrics = [0.360, 0.829, 0.845, 0.412]
    # region_colors = [get_dice_color(dice) for dice in metrics]
    # create_seg_stick(patient_name, region_colors, STICK_BASE_PNG, STICK_BASE_REGIONS, "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/seg_sticks")
    #
    # patient_name = 'AUTOMI_average_female'
    # metrics = [0.339, 0.791, 0.832, 0.389]
    # region_colors = [get_dice_color(dice) for dice in metrics]
    # create_seg_stick(patient_name, region_colors, STICK_BASE_PNG, STICK_BASE_REGIONS,
    #                  "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/seg_sticks")

    #Create the segmentation sticks for the dataset for HD95
    # create_region_metrics_and_segsticks_dataset("/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/labelsTr",
    #                          "/home/aiart/nnUnetAUTOMI/dataset/nnUNet_results/Dataset003_AUTOMI_CTVLNF_NEWGL/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/",
    #                          "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/seg_sticks_hd95",
    #                          "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/patient_regions.json",
    #                          "hd95")



    # EXPERIMENT 2 - CTV MULTIPLE INPUTS 2 DATASET 9
    # Create the segmentation sticks for the dataset
    create_region_metrics_and_segsticks_dataset(
        "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset009_AUTOMI_CTV_1_multiple_inputs/labelsTr",
        "/home/aiart/nnUnetAUTOMI/dataset/nnUNet_results/Dataset009_AUTOMI_CTV_1_multiple_inputs/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/",
        "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset009_AUTOMI_CTV_1_multiple_inputs/seg_sticks",
        "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/patient_regions.json",
        "dice")

    # EXPERIMENT 3 - CTV MULTIPLE INPUTS 2 DATASET 10
    #Create the segmentation sticks for the dataset
    create_region_metrics_and_segsticks_dataset("/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset010_AUTOMI_CTV_2_multiple_inputs/labelsTr",
                             "/home/aiart/nnUnetAUTOMI/dataset/nnUNet_results/Dataset010_AUTOMI_CTV_2_multiple_inputs/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/",
                             "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset010_AUTOMI_CTV_2_multiple_inputs/seg_sticks",
                             "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset010_AUTOMI_CTV_2_multiple_inputs/patient_regions.json",
                             "dice")

    # EXPERIMENT 3 - CTV MULTIPLE INPUTS 1 DATASET 5
    # Create the segmentation sticks for the dataset
    create_region_metrics_and_segsticks_dataset(
        "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset005_AUTOMI_CTV_1_multiple_inputs/labelsTr",
        "/home/aiart/nnUnetAUTOMI/dataset/nnUNet_results/Dataset005_AUTOMI_CTV_1_multiple_inputs/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessed",
        "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset005_AUTOMI_CTV_1_multiple_inputs/seg_sticks",
        "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/patient_regions.json",
        "dice")

    # EXPERIMENT 4 - CTV MULTIPLE INPUTS 2 DATASET 6
    # Create the segmentation sticks for the dataset
    create_region_metrics_and_segsticks_dataset(
        "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset006_AUTOMI_CTV_2_multiple_inputs/labelsTr",
        "/home/aiart/nnUnetAUTOMI/dataset/nnUNet_results/Dataset006_AUTOMI_CTV_2_multiple_inputs/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/",
        "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset006_AUTOMI_CTV_2_multiple_inputs/seg_sticks",
        "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/patient_regions.json",
        "dice")

    # EXPERIMENT 5 - CTV MULTIPLE INPUTS 3 DATASET 7
    # Create the segmentation sticks for the dataset
    create_region_metrics_and_segsticks_dataset(
        "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset007_AUTOMI_CTV_3_multiple_inputs/labelsTr",
        "/home/aiart/nnUnetAUTOMI/dataset/nnUNet_results/Dataset007_AUTOMI_CTV_3_multiple_inputs/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/",
        "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset007_AUTOMI_CTV_3_multiple_inputs/seg_sticks",
        "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/patient_regions.json",
        "dice")

    # SegSticks for the Gender difference table
    # 6 segsticks for for each fairness metric, agd, mgd, dq

    # # AGD
    # segstick_names = ['agd_base_segstick', 'agd_mi_z_segstick', 'agd_ei_z_segstick', 'agd_crop_z_segstick', 'agd_mi_segstick', 'agd_mi_ts_segstick']
    # segstick_metrics = [[0.021	,0.038,	0.012,	0.023],
    #                     [0.027	,0.036,	0.011,	0.028],
    #                     [0.027,	0.038	,0.010,	0.025],
    #                     [0.028,	0.037	,0.012	,0.027],
    #                     [0.020	,0.038,	0.016,	0.024],
    #                     [0.017,	0.036,	0.013,	0.029]]
    #
    # for i, metrics in enumerate(segstick_metrics):
    #     region_colors = [get_dice_color(dice, norm='agd') for dice in metrics]
    #     create_seg_stick(segstick_names[i], region_colors, STICK_BASE_PNG, STICK_BASE_REGIONS,
    #                      "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/seg_sticks")
    #
    # # MGD
    # segstick_names = ['mgd_base_segstick', 'mgd_mi_z_segstick', 'mgd_ei_z_segstick', 'mgd_crop_z_segstick', 'mgd_mi_segstick', 'mgd_mi_ts_segstick']
    #
    # segstick_metrics = [[0.022, 0.069,	0.027,	0.033],
    #                     [0.019,	0.064,	0.005,	0.026],
    #                     [0.019,	0.073,	0.001,	0.025],
    #                     [0.019,	0.061,	0.010,	0.027],
    #                     [0.020,	0.067,	0.011,	0.037],
    #                     [0.025,	0.056,	0.026,	0.035]]
    #
    # for i, metrics in enumerate(segstick_metrics):
    #     region_colors = [get_dice_color(dice, norm='mdg') for dice in metrics]
    #     create_seg_stick(segstick_names[i], region_colors, STICK_BASE_PNG, STICK_BASE_REGIONS,
    #                      "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/seg_sticks")
    #
    # # QD
    # segstick_names = ['dq_base_segstick', 'dq_mi_z_segstick', 'dq_ei_z_segstick', 'dq_crop_z_segstick', 'dq_mi_segstick', 'dq_mi_ts_segstick']
    #
    # segstick_metrics = [[0.091,	0.111,	0.091,	0.136],
    #                     [0.094,	0.133,	0.109,	0.126],
    #                     [0.092,	0.141,	0.102,	0.113],
    #                     [0.094,	0.137,	0.098,	0.121],
    #                     [0.086,	0.133,	0.100,	0.133],
    #                     [0.085,	0.116,	0.099,	0.134]]
    #
    # for i, metrics in enumerate(segstick_metrics):
    #     region_colors = [get_dice_color(dice, norm ='qd') for dice in metrics]
    #     create_seg_stick(segstick_names[i], region_colors, STICK_BASE_PNG, STICK_BASE_REGIONS,
    #                      "/home/aiart/nnUnetAUTOMI/dataset/nnUnet_raw/Dataset003_AUTOMI_CTVLNF_NEWGL/seg_sticks")
