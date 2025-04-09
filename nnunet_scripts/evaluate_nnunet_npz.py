import os, sys
import csv
import pydicom
import pandas as pd
import numpy as np
import seg_metrics.seg_metrics as sg
import argparse

def binary_compare(predPath, spacing, metrics=['dice'], target_normalized=False):
    try:
        arrayNPZ = np.load(predPath)
        pred = arrayNPZ['pred']
        target = arrayNPZ['target']
        pred = np.round(pred / np.max(pred)).astype("int8")
        if target_normalized:
            target = np.round(target / np.max(target)).astype("int8")
        else:
            target = np.round(target / 255).astype("int8")

    except Exception:
        print("File not existing")
        return None
    if target.shape == pred.shape:
        return sg.write_metrics(labels=[1], gdth_img=target, pred_img=pred, csv_file=None,  spacing=spacing, metrics=metrics)
    else:
        print(f'Shape mismatch. pred={str(pred.shape)} and gt={str(target.shape)}')
        return None

def evaluateDatasetNPZ_nnunet(predictions_path, patient_dictionary_csv, original_dataset_path, evaluation_xsl_path, structure, metrics=['dice', 'hd', 'hd95']):
    results = pd.DataFrame(columns=['patient', 'pred', 'target', 'dice', 'hd', 'hd95'])

    for i, patientFile in enumerate(os.listdir(predictions_path)):
        print(patientFile)
        patientPath = os.path.join(predictions_path, patientFile)
        patientName_nnunet = patientFile.split('.')[0]

        # Read the mapping from old names to new names from the second CSV file
        name_mapping = {}
        with open(patient_dictionary_csv, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                old_name = row[0]
                new_name = row[1]
                name_mapping[new_name] = old_name

        # Get the key of patientName_nnunet
        patientName = name_mapping[patientName_nnunet]

        patientOriginalPath = os.path.join(original_dataset_path, 'imgs', patientName)
        listImages = os.listdir(patientOriginalPath)
        firstImagePath = os.path.join(patientOriginalPath, listImages[0])
        ds = pydicom.dcmread(firstImagePath)

        st = ds.SliceThickness
        ps = ds.PixelSpacing
        spacing = [st, ps[0], ps[1]]
        res = binary_compare(patientPath, metrics=metrics, spacing=spacing, target_normalized=True)

        if res is not None:
            patientResults = {'patient': patientName , 'pred': structure, 'target': structure,
                              'dice': res[0]['dice'][0], 'hd': res[0]['hd'][0], 'hd95': res[0]['hd95'][0]}
            print(res[0]['dice'][0], res[0]['hd'][0], res[0]['hd95'][0])
            results = pd.concat([results, pd.DataFrame(patientResults, index=[i])], ignore_index=True)

    results.to_excel(evaluation_xsl_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the nnunet predictions')
    parser.add_argument('--pred_folder', type=str, required=True, help='Path to the folder with the predictions')
    parser.add_argument('--patient_dictionary_csv', type=str, required=True, help='Path to the patient dictionary csv file')
    parser.add_argument('--img_path', type=str, required=True, help='Path to the folder with the original images')
    parser.add_argument('--evaluation_path', type=str, required=True, help='Path to the folder and excel file where the evaluation will be saved')
    parser.add_argument('--structure', type=str, required=True, help='Name of the structure being evaluated (Example: PTV_tot_pred)')
    args = parser.parse_args()

    evaluateDatasetNPZ_nnunet(args.pred_folder, args.patient_dictionary_csv, args.img_path, args.evaluation_path, args.structure)

