''' This script is dedicated to organise the dataset AUTOMI in a way to better read and be used by the machine learning
pipeline.'''

import os, sys
import csv
from PIL import Image
from matplotlib import pyplot as plt
import datapaths
import pydicom
import utils_data

if __name__ == "__main__":

    datapath = datapaths.datapaths['AUTOMI']
    originalImgpath = os.path.join(datapath, 'ExportDec2021')
    ctOrientationPath = os.path.join(datapath, 'ctOrientation.json')

    imgPath = os.path.join(datapath, 'imgs')
    if not os.path.exists(imgPath):
        os.mkdir(imgPath)
    maskPath = os.path.join(datapath, 'masks')
    if not os.path.exists(maskPath):
        os.mkdir(maskPath)

    for patient in os.listdir(originalImgpath):
        originalPatientPath = os.path.join(originalImgpath, patient)
        patientPath = os.path.join(imgPath, patient)
        if not os.path.exists(patientPath):
            os.mkdir(patientPath)
        patientMaskPath = os.path.join(maskPath, patient)
        if not os.path.exists(patientMaskPath):
            os.mkdir(patientMaskPath)

        # Check if the order of the slices of the CT scan corresponds with the order of the RTStruct
        isPatientOriented = utils_data.isPatientOriented(patient, ctOrientationPath)

        # Moves the .dcm images from the original patient path to the new path and renames the file with
        # the correct slice number
        utils_data.moveRenameAllDicom(originalPatientPath, patientPath, isPatientOriented)

        # Finds the path for the RTStruct containing the grountruth masks and turns them into .npy arrays in the new folder
        rtStructFilename = utils_data.getPatientRTStructFileName(originalPatientPath)
        utils_data.convertRtStructTo3dArrays(originalPatientPath, patientMaskPath)






