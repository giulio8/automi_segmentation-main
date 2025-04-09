'''
This script turns the .npz files into .dcm RTSTRUCT files that can be read by 3D Slicer, for example.
'''

import os, sys
import utils_data
import datapaths
import numpy as np
import utils_evaluation

def main():
    modelFolder = os.path.join(datapaths.resultspath, 'AUTOMI_PTV_tot_BCELoss')
    structure = 'PTV_tot_pred'
    evaluationSet = 'AUTOMI_test'
    foldName = 'fold4predictionsNPZ'
    foldRTStructFolderName = 'fold4predictionsRTSTRUCT'
    predictionsFolder = os.path.join(modelFolder, evaluationSet)
    foldFolder = os.path.join(predictionsFolder, foldName)
    foldRTStructFolder = os.path.join(predictionsFolder, foldRTStructFolderName)

    if not os.path.isdir(foldRTStructFolder):
        os.mkdir(foldRTStructFolder)

    for patient_filename in os.listdir(foldFolder):
        print(f'Patient name: {patient_filename}')

        patient_name = patient_filename.split('.')[0]
        patientTrainingDicomPath = os.path.join(datapaths.datapaths['AUTOMI'], 'imgs', patient_name)
        patientOriginalDicomPath = os.path.join(datapaths.datapaths['AUTOMI'], 'ExportDec2021', patient_name)
        patientOriginalRTStructName = utils_data.getPatientRTStructFileName(patientOriginalDicomPath)
        patientOriginalRTStructPath = os.path.join(patientOriginalDicomPath, patientOriginalRTStructName)

        print(f'Patient DICOM Datapath: {patientTrainingDicomPath}')
        print(f'Patient No. of Slices: {len(os.listdir(patientTrainingDicomPath))}')

        # Gets the prediction array from the .npz file
        arrayNPZ = np.load(os.path.join(foldFolder, patient_filename))
        pred = arrayNPZ['pred']

        pred = utils_evaluation.post_process_prediction(pred)

        # Saves the prediction array as a .dcm RTSTRUCT file containing the as well the groundtruth
        utils_data.savePredsNPZtoRTStruct(pred, patient_name, structure,
                                          patientTrainingDicomPath,
                                          foldRTStructFolder,
                                          roi_number=100,
                                          patientOriginalRTStructPath=patientOriginalRTStructPath)


if __name__ == '__main__':
    main()