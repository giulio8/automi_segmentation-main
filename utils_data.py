import os
import re
import json
import pandas as pd
import pydicom
import numpy as np
import nibabel as nib
from rt_utils import RTStructBuilder
from PIL import Image
import shutil
import torch.utils.data as data
from torchvision import transforms
import utils_evaluation
import SimpleITK as sitk



def convertRTStructToPng(patientPath, newPatientPath):
    rtStructName = getPatientRTStructFileName(patientPath)
    rtStructPath = os.path.join(patientPath, rtStructName)

    patientName = os.path.split(patientPath)[-1]
    # Load existing RT Struct. Requires RT Struct name
    rtstruct = RTStructBuilder.create_from(dicom_series_path=patientPath, rt_struct_path=rtStructPath)

    # View all of the ROI names from within the image
    roiList = rtstruct.get_roi_names()
    print(roiList)

    # Loading the 3D Mask from within the RT Struct for each ROI
    for roi in roiList:
        try:
            roi3dMask = rtstruct.get_roi_mask_by_name(roi)
        except:
            print('ROI: \'{}\' does not have a Contour Sequence!'.format(roi))

        roiFolderPath = os.path.join(newPatientPath, roi)
        if not os.path.exists(roiFolderPath):
            os.makedirs(roiFolderPath)

        numSlices = roi3dMask.shape[2]
        for numSlice in range(numSlices):
            roi2dMask = roi3dMask[:, :, numSlice]
            roi2dMask = Image.fromarray(roi2dMask)
            roi2dMaskName = patientName + '_' + str(numSlice).zfill(3) + '.png'
            roi2dMask.save(os.path.join(roiFolderPath, roi2dMaskName))


def convertRtStructTo3dArrays(patientPath, newPatientPath):
    rtStructName = getPatientRTStructFileName(patientPath)
    rtStructPath = os.path.join(patientPath, rtStructName)

    patientName = os.path.split(patientPath)[-1]
    # Load existing RT Struct. Requires RT Struct name
    rtstruct = RTStructBuilder.create_from(dicom_series_path=patientPath, rt_struct_path=rtStructPath)

    # Displays all of the ROI names from within the image
    roiList = rtstruct.get_roi_names()
    print(roiList)

    for roi in roiList:
        try:
            roi3dMask = rtstruct.get_roi_mask_by_name(roi)
            roi3dMask = roi3dMask > 0.5
            roi3dMask = roi3dMask.astype(np.uint8)
            # print(type(roi3dMask[1]))
            roiPath = os.path.join(newPatientPath, roi)
            np.savez_compressed(roiPath, roi3dMask)
        except:
            print('ROI: \'{}\' does not have a Contour Sequence!'.format(roi))


def getPatientRTStructFileName(patientPath, subString='RTSTRUCT'):
    patientRTStructFileName = None
    for fileName in os.listdir(patientPath):
        if subString in fileName:
            patientRTStructFileName = fileName

    return patientRTStructFileName


def convertCTDicomToPNG(dicomPath):
    dicom = pydicom.dcmread(dicomPath)
    instanceNumber = dicom.InstanceNumber - 1

    pixelArray = dicom.pixel_array.astype(float)
    scaledImage = (np.maximum(pixelArray, 0) / pixelArray.max()) * 255.0
    scaledImage = np.uint8(scaledImage)
    finalImage = Image.fromarray(scaledImage)

    return instanceNumber, finalImage


def convertAllDicomAutomiToPNG(patientPath, newPatientPath, isPatientOriented):
    RTStructSubString = 'RTSTRUCT'
    RTDoseSubString = 'RTDOSE'

    numTotalDicoms = 0
    for fileName in os.listdir(patientPath):
        if not RTStructSubString in fileName and not RTDoseSubString in fileName:
            numTotalDicoms += 1

    for fileName in os.listdir(patientPath):
        if not RTStructSubString in fileName and not RTDoseSubString in fileName:
            sliceNumber, pngSlice = convertCTDicomToPNG(os.path.join(patientPath, fileName))

            if isPatientOriented:
                sliceNumber = sliceNumber
            else:
                sliceNumber = numTotalDicoms - sliceNumber - 1

            pngName = os.path.split(patientPath)[-1] + '_' + str(sliceNumber).zfill(3) + '.png'
            pngSlice.save(os.path.join(newPatientPath, pngName))


def moveRenameAllDicom(patientPath, newPatientPath, isPatientOriented):
    RTStructSubString = 'RTSTRUCT'
    RTDoseSubString = 'RTDOSE'

    numTotalDicoms = 0
    for fileName in os.listdir(patientPath):
        if not RTStructSubString in fileName and not RTDoseSubString in fileName:
            numTotalDicoms += 1

    for fileName in os.listdir(patientPath):
        if not RTStructSubString in fileName and not RTDoseSubString in fileName:
            sliceNumber, pngSlice = convertCTDicomToPNG(os.path.join(patientPath, fileName))

            if isPatientOriented:
                sliceNumber = sliceNumber
            else:
                sliceNumber = numTotalDicoms - sliceNumber - 1

            newFileName = os.path.split(patientPath)[-1] + '_' + str(sliceNumber).zfill(3) + '.dcm'
            shutil.copy(os.path.join(patientPath, fileName), os.path.join(newPatientPath, newFileName))
            # newFileName = os.path.split(patientPath)[-1] + '_' + str(sliceNumber).zfill(3) + '.png'
            #pngSlice.save(os.path.join(newPatientPath, newFileName))


def fold_distribution_automi(csv_fold_dist_path, img_path):
    '''
    Uses a .csv that contains the patientID and fold number and creates a .csv that contains the images path of that
    patient and the subset (train, val, test) that the image belongs to.
    Then it saves the .csv of that fold distribution in the AUTOMI dataset folder.
    If the fold is 1 then the fold distribution is train: 1,2,3 , val: 4, test: 5
    :param csv_fold_dist_path:
    :param img_path: dicom images path with folders of the patients
    :return:
    '''

    fold_dist = [[1,2,3,4,5], [5,1,2,3,4], [4,5,1,2,3], [3,4,5,1,2], [2,3,4,5,1]]
    patientDistDF = pd.read_csv(csv_fold_dist_path, usecols=['patientID', 'fold'], delimiter=',')

    for fold in range(1, 6):
        data = []
        for ind in patientDistDF.index:
            patient = patientDistDF['patientID'][ind]
            foldNum = patientDistDF['fold'][ind]
            patientImgPath = os.path.join(img_path, 'imgs', patient)
            subset = 'train'

            if foldNum == fold_dist[fold-1][0] or foldNum == fold_dist[fold-1][1] or foldNum == fold_dist[fold-1][2]:
                subset = 'train'
            elif foldNum == fold_dist[fold-1][3]:
                subset = 'val'
            elif foldNum == fold_dist[fold-1][4]:
                subset = 'test'

            for image in os.listdir(patientImgPath):
                imagePath = os.path.join(patientImgPath, image)

                data.append([imagePath, subset])

        data.sort()

        imageDistDF = pd.DataFrame(data, columns=['imagePath', 'subset'])
        imageDistDF.to_csv(os.path.join(img_path, f'imageDist_fold{fold}.csv'))


def savePredsNPZtoRTStruct(pred_3d_array, patient_name, structure, original_dicom_path, results_path, roi_number, original_rt_struct_path=None):
    if original_rt_struct_path is None:
        rt_struct = RTStructBuilder.create_new(dicom_series_path=original_dicom_path)
    else:
        rt_struct = RTStructBuilder.create_from(dicom_series_path=original_dicom_path,
                                                  rt_struct_path=original_rt_struct_path)

    if isinstance(pred_3d_array, list) and isinstance(structure, list):
        for pred, struct in zip(pred_3d_array, structure):
            rt_struct.add_roi(mask=pred, name=struct, use_pin_hole=True, roi_number=roi_number)
            roi_number += 1
        rt_struct.save(os.path.join(results_path, f'{patient_name}_new_rt_struct.dcm'))
    else:
        rt_struct.add_roi(mask=pred_3d_array, name=structure, use_pin_hole=True, roi_number=roi_number)
        rt_struct.save(os.path.join(results_path, f'{patient_name}_{structure}.dcm'))


def predsTo3dArray(patientPredsPathList):
    '''
    This function takes a list of paths to the predictions of a patient and returns a 3d array with the predictions
    :param patientPredsPathList:
    :return:
    '''
    # Orders the patientPredsPathList
    patientPred3dArray = []

    # Reads the list of predictions
    for i, predPath in enumerate(patientPredsPathList):
        pred = Image.open(predPath)
        predArray = np.array(pred)
        patientPred3dArray.append(predArray) # Adds the prediction to the 3d array

    return patientPred3dArray


def convert_png_to_npz(prediction_png_path, patient_name, npz_result_path):
    # Gets the pred png and the target png paths from the patient's prediction folder
    patientPredPaths = getPatientPreds(prediction_png_path, patient_name, maskString='mask')
    patientTargetPaths = getPatientPreds(prediction_png_path, patient_name, maskString='target')

    # Turns the pred png and the target png into two 3d arrays
    patientPred3Darray = predsTo3dArray(patientPredPaths)
    patientTarget3Darray = predsTo3dArray(patientTargetPaths)


    np.savez_compressed(os.path.join(npz_result_path, patient_name), pred=patientPred3Darray, target=patientTarget3Darray)


def convert_dicom_to_nifti(input, output):
    '''
    This function converts a DICOM series image to a NIfTI image file.
    :param input: folder of the DICOM series
    :param output:
    :return:
    '''
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(input)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, output)


def convert_npz_to_nii(npz_path, ct_folder_path, output_file):
    '''
    This function transforms the .npz files that contain the binary masks of a certain structure and convert them to a NIfTI image.
    Note: But the affine transformation is not correct.
    :param folder_path:
    :param structure:
    :return:
    '''

    # Find the corresponding CT files in the folder
    ct_files = [f for f in os.listdir(ct_folder_path) if f.endswith('.dcm')]

    # Sort the CT files based on the Image Position Patient (IPP) value
    ct_files = sorted(ct_files, key=lambda x: pydicom.dcmread(os.path.join(ct_folder_path, x)).ImagePositionPatient[2])

    # Load the first CT file to get the necessary information
    first_ct = pydicom.dcmread(os.path.join(ct_folder_path, ct_files[0]))
    second_ct = pydicom.dcmread(os.path.join(ct_folder_path, ct_files[1]))

    # Load the .npz file
    arr = np.load(npz_path)
    mask3d = arr['arr_0']
    print(f'Shape of the 3dmask: {mask3d.shape}')

    sitk_mask = sitk.GetImageFromArray(mask3d)

    # Set the origin, spacing, and direction from the CT image
    sitk_mask.SetOrigin(first_ct.ImagePositionPatient)
    sitk_spacing = [float(sp) for sp in first_ct.PixelSpacing]
    # Calculate the spacing between slices
    spacing_between_slices = (float(second_ct.ImagePositionPatient[2]) - float(first_ct.ImagePositionPatient[2]) + float(first_ct.SliceThickness))

    sitk_spacing.append(spacing_between_slices)
    sitk_mask.SetSpacing(sitk_spacing)
    # Calculate the direction of the image

    sitk_mask.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])

    # Save the mask as a NIfTI file
    sitk.WriteImage(sitk_mask, output_file)

def prediction_to_rtstruct(original_dicom_path, predictions_path, result_rtstruct_folder_path, patient_name,
                           structure, roi_number, prediction_format='npz', original_rt_struct_path=None):
    '''
    This function takes the predictions of a patient and saves them as RTStruct files.
    :param original_dicom_path: path to the original DICOM images folder of the patient
    :param predictions_path: path to the original predictions folder, in the case of the .png format it's a folder per patient,
    in case of the .npz and .nii.gz format it's a folder with all the predictions.
    :param result_rtstruct_folder_path: result path where the RTStruct files are going to be saved
    :param patient_name: name of the patient
    :param structure: structure name to be saved in the RTStruct
    :param roi_number: index of the roi that is going to be added to the RTStruct, has to be different from all the other indexes, or it can be added
    Note: there is a possibility that for example: Index 52 is available and then 53 was already occupied.
    :param prediction_format: file format of the predictions, can be "png", "npz" or "nii"
    :param original_rt_struct_path: original path of the RTStruct of that patient
    :return:
    '''
    print(f'Patient name: {patient_name}')

    pred = None

    # If the predictions are in .png format, transform them into a 3D array
    if prediction_format == 'png':
        print('Turning the predictions from .png format to 3D array')
        pred = predsTo3dArray(os.path.join(predictions_path, patient_name))
    elif prediction_format == 'npz':
        # Gets the prediction array from the .npz file
        print('Turning the predictions from .npz format to 3D array')
        arrayNPZ = np.load(os.path.join(predictions_path, f'{patient_name}.npz'))
        pred = arrayNPZ['pred']
    elif prediction_format == 'nii':
        # Gets the prediction array from the .nii.gz file
        print('Turning the predictions from .nii.gz format to 3D array')
        pred = nib.load(os.path.join(predictions_path, f'{patient_name}.nii.gz')).get_fdata()

    if prediction_format != 'nii':
        # Post-process the predictions
        pred = utils_evaluation.post_process_prediction(pred)

    # Save the predictions as RTStruct
    savePredsNPZtoRTStruct(pred, patient_name, structure,
                                      original_dicom_path,
                                      result_rtstruct_folder_path,
                                      roi_number=roi_number,
                                      original_rt_struct_path=original_rt_struct_path)


def isPatientOriented(patientName, ctOrientationFilePath):
    '''
  This function uses an existing JSON file that contains information regarding each patient's CT scan orientation and
  checks retrieves the boolean value of the patient's orientation. If it is true, it is correctly oriented if it is false,
  it means it has to be flipped.
  :param patientName: Code that identifies the patient.
  :param ctOrientationFilePath: Path of the JSON file.
  :return: Boolean value of patient's orientation.
  '''
    f = open(ctOrientationFilePath)
    data = json.load(f)
    f.close()

    isPatientOriented = data[patientName]

    return isPatientOriented


def cleanLabels(datasetPath, structureListEnglish, structureListItalian):
    for patient in os.listdir(datasetPath):
        patientPath = os.path.join(datasetPath, patient, 'MAN')

        newPatientPath = os.path.join(datasetPath, patient, 'MAN_cleaned')
        if not os.path.exists(newPatientPath):
            os.makedirs(newPatientPath)

        for roiFile in os.listdir(patientPath):
            roi = roiFile.split('.')[0]
            roi = re.sub('[^A-Za-z0-9]+', '', roi)
            roi = roi.lower()
            for i, structIT in enumerate(structureListItalian):
                if structIT in roi:
                    roiPath = os.path.join(patientPath, roiFile)
                    roiFilenameEnglish = structureListEnglish[i] + '.npy'
                    newRoiPath = os.path.join(newPatientPath, roiFilenameEnglish)
                    shutil.copy(roiPath, newRoiPath)


def flipPreds(datasetPath, ctOrientationFilePath):
    '''
    This function flips the predictions of the patients that are not correctly oriented based in the ctOrientation .csv file.
    :param datasetPath:
    :param ctOrientationFilePath:
    :return:
    '''
    for patient in os.listdir(datasetPath):
        patientPathPOLI = os.path.join(datasetPath, patient, 'POLI')
        patientOriented = isPatientOriented(patient, ctOrientationFilePath)
        if patientOriented:
            print('{}: Oriented'.format(patient))
        else:
            print('{}: Not Oriented'.format(patient))
        if not patientOriented:
            for struct in os.listdir(patientPathPOLI):
                structPath = os.path.join(patientPathPOLI, struct)
                structArray = np.load(structPath)
                structArray = np.flip(structArray, axis=2)
                print('{} was flipped'.format(struct))
                np.save(structPath, structArray)


def getIndexes(csvPath, subset='train'):
    '''
    This function reads the csv file containing the indexes of the patients that are going to be used for training, validation
    or test set.
    :param csvPath:
    :param subset:
    :return:
    '''
    imageDist = pd.read_csv(csvPath)
    indexList = []

    for ind in imageDist.index:

        if imageDist['subset'][ind] == subset:
            indexList.append(ind)

    return indexList


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        try:
            if np.max(img) > 255:
                imgdata = np.asarray(img)
                imgdata = (imgdata + 1) / 256 - 1
                img = Image.fromarray(np.uint8(imgdata))
        except:
            raise ('Error in image {}'.format(path))
        return img.convert('L')


def CropMinBorders(img,tgt):
    imgB = np.array(img) != np.min(img)
    imgBx = np.nonzero(np.sum(imgB,axis=0))
    imgBy = np.nonzero(np.sum(imgB, axis=1))

    xmin = np.min(imgBx)
    ymin = np.min(imgBy)
    xmax = np.max(imgBx)
    ymax = np.max(imgBy)

    img = img.crop((xmin, ymin, xmax+1, ymax+1))

    # If the tgt is a list (of targets) this means it's a multi-class segmentation problem
    if isinstance(tgt,list):
        newTgtList = []
        for target in tgt:
            target = target.crop((xmin, ymin, xmax + 1, ymax + 1))
            newTgtList.append(target)
        tgt = newTgtList
    else:
        tgt = tgt.crop((xmin, ymin, xmax+1, ymax+1))

    return img, tgt


class DatasetFolderAutomi(data.Dataset):

    def __init__(self, datasetPath, structure, transform=None, datasetCSV=None):
        '''
    In this dataset class, the "init" function will read and store in data, all the images from the AUTOMI dataset
    separated by patient and all the masks that are going to be used for the training. The name of the classes are
    specified in the classList.
    '''

        self.datasetPath = datasetPath
        self.structure = structure
        self.transform = transform
        self.datasetCSV = datasetCSV

        self.allImagesPaths = []
        self.allMasksPath = []

        imgDatasetPath = os.path.join(datasetPath, 'imgs')
        masksDatasetPath = os.path.join(datasetPath, 'masks')

        for patient in os.listdir(imgDatasetPath):

            patientImgPath = os.path.join(imgDatasetPath, patient)
            for image in os.listdir(patientImgPath):
                imagePath = os.path.join(patientImgPath, image)
                self.allImagesPaths.append(imagePath)

            if self.structure is not None:
                patientMaskPath = os.path.join(masksDatasetPath, patient)
                structureFileName = structure + '.npz'
                structurePath = os.path.join(patientMaskPath, structureFileName)
                self.allMasksPath.append(structurePath)

        self.allImagesPaths.sort()
        self.allMasksPath.sort()

    def __len__(self):
        return len(self.allImagesPaths)

    def __getitem__(self, index):

        # Gets image from all the image list and determines the patient name and slice number.
        imagePath = self.allImagesPaths[index]
        patientName = imagePath.split('/')[-2]
        sliceNumber = imagePath.split('/')[-1]
        sliceNumber = sliceNumber.split('_')[-1]
        sliceNumber = int(sliceNumber.split('.')[0])

        strSliceNumber = str(sliceNumber)
        fName = f'{patientName}_{strSliceNumber.zfill(3)}.npy'

        # The image is loaded using that path
        image = pydicom.dcmread(imagePath)


        if self.structure is not None:
            # For all patients masks gets the index of the mask that has that patient and structure name
            mask3dStructIndexes = [i for i, path in enumerate(self.allMasksPath) if
                                   patientName in path and self.structure in path]
            maskPath = self.allMasksPath[mask3dStructIndexes[0]]

            # The mask is loaded and converted in to an int8 format.
            try:
                arr = np.load(maskPath)
                masks3d = arr['arr_0']
                mask = masks3d[:, :, sliceNumber].astype(np.uint8) * 255
            except:
                mask = image.pixel_array.astype(np.uint8)
                mask = mask * 255
        else:
            mask = np.zeros(image.pixel_array.shape).astype(np.uint8)

        # if transforms are enabled, both the image and the mask are transformed.
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            image = image.pixel_array.astype(np.uint8)

            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)

        return image, mask, fName


class TrainTransforms(object):
    def __init__(self, modelParams, toTensor=True):
        self.sizeInput = modelParams['size_input']
        self.toTensor = toTensor

    def __call__(self, image, mask):
        image = imageToHU(image)
        image = transformLUT(image, 40, 400).astype(np.uint8)
        # image = image.pixel_array
        # image = (image / np.max(image) * 255).astype('uint8')
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        # Useful for debugging the transforms.
        # plt.figure()
        # plt.imshow(image)
        # plt.title('Image')
        # plt.figure()
        # plt.imshow(mask)
        # plt.title('Mask')
        # plt.show()

        transfResize = transforms.Resize(size=self.sizeInput)
        image = transfResize(image)
        mask = transfResize(mask)

        # transfRotationAngle = transforms.RandomRotation(degrees=(10,10))
        #
        # image = transfRotation(image)
        # mask = transfRotation(mask)

        transfColorJitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1)
        image = transfColorJitter(image)

        # Useful for debugging the transforms
        # plt.figure()
        # plt.imshow(image)
        # plt.title('Image Transformed')
        # plt.figure()
        # plt.imshow(mask)
        # plt.title('Mask Transformed')
        # plt.show()

        if self.toTensor:
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)

        return image, mask


class TestTransform:
    def __init__(self, modelParams, toTensor=True):
        self.sizeInput = modelParams['size_input']
        self.toTensor = toTensor

    def __call__(self, image, mask):
        image = imageToHU(image)
        image = transformLUT(image, 40, 400).astype(np.uint8)
        # image = image.pixel_array
        # image = (image / np.max(image) * 255).astype('uint8')

        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        transfResize = transforms.Resize(size=self.sizeInput)
        image = transfResize(image)
        mask = transfResize(mask)

        if self.toTensor:
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)

        return image, mask


def imageToHU(image):
    '''
  Transforms a dycom image into an array in terms of Hounsfield Units (HU).
  :param image: dycom image
  :return: image array in HU
  '''
    intercept = image.RescaleIntercept
    slope = image.RescaleSlope
    imageHU = image.pixel_array * slope + intercept

    return imageHU


def transformLUT(image, center, width):
    '''
  Uses a Look Up Table to transform the image in to a more useful Hounsfield Unit range.
  The image is then normalized between values of 0 and 255 (8 bits).
  :param image: image array in Hounsfield Units
  :param center:
  :param width:
  :return:
  '''

    imgMin = center - width // 2
    imgMax = center + width // 2
    windowImage = image.copy()
    windowImage[windowImage < imgMin] = imgMin
    windowImage[windowImage > imgMax] = imgMax

    windowImage = (255 * (windowImage - imgMin)) // width

    return windowImage


def getPatientDict(foldFolder, patientNames):
    '''
    Creates a dictionary with the patient names as keys and the path to the patient folder as values.
    :param foldFolder:
    :param patientNames:
    :return:
    '''
    patientDict = {}
    for patient in patientNames:
        for file in os.listdir(foldFolder):
            if patient in file:
                patientDict[patient] = os.path.join(foldFolder, file)
    return patientDict


def getPatientNames(foldFolder):
    ''' This function returns a list with the patient names in the fold folder. '''
    patientNames = []
    for file in os.listdir(foldFolder):
        patientName = file.split('_')[0]
        if patientName not in patientNames:
            patientNames.append(patientName)
    return patientNames

def getPatientPreds(foldFolder, patientName, maskString='mask'):
    '''
    This function returns a list with the paths to the masks of the patient.
    :param foldFolder:
    :param patientName:
    :return:
    '''
    patientPreds = []
    for file in os.listdir(foldFolder):
        if patientName in file and maskString in file:
            patientPreds.append(os.path.join(foldFolder, file))

    patientPreds.sort()
    return patientPreds
