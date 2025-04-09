from scipy import stats, ndimage
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as ssim
from skimage import morphology
import os, sys
import numpy as np
import pydicom
from rt_utils import RTStructBuilder
import seg_metrics.seg_metrics as sg
import pandas as pd


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

def getDistMet(tgt, pred, func):
    distMet = []
    for t,p in zip(tgt,pred):
        t = np.round(t/255)
        p = np.round(p/np.max(p))
        distMet.append(func(t,p))
    return distMet


def calcIoU(t, p):
    return np.sum(np.logical_and(t,p))/np.sum(np.logical_or(t,p))


def calcDiceScore(t, p):
    return 2 * np.sum(np.logical_and(t,p))/(np.sum(t) + np.sum(p))


def calcJaccardIndex(t, p):
    union = ((t + p) > 0).sum()
    inter = ((t + p) == 2).sum()

    jaccardIndex = inter / union

    return jaccardIndex


def calcHausdorffMAD(target, prediction):
    struct = ndimage.generate_binary_structure(2, 1)
    erode = ndimage.binary_erosion(target, struct)
    gtEdge = (target>0) ^ erode
    erode = ndimage.binary_erosion(prediction, struct)
    pdEdge = (prediction>0) ^ erode

    c1 = np.array(np.where(gtEdge == 1)).transpose()
    c2 = np.array(np.where(pdEdge == 1)).transpose()

    d = cdist(c1, c2)
    d1 = np.amin(d, axis=0)
    d2 = np.amin(d, axis=1)

    hausdorff = max(max(d1), max(d2))
    mad = sum([sum(d1) / len(d1), sum(d2) / len(d2)]) / 2
    # Calculate HD95 using np.percentile
    hd95 = np.percentile(np.concatenate([d1, d2]), 95)

    return hausdorff, mad, hd95

def hausdorff_exclusion(target, prediction, exclusion):
    '''
    This is a modified version of the hausdorff distance that excludes points that are in the surface of the exclusion
    and the surface of the prediction
    :param target: binary mask of the groundtruth target after the exclusion of another mask called "exclusion"
    :param prediction: binary mask of the prediction after the exclusion of another mask called "exclusion"
    :param exclusion: binary mask of the exclusion area.
    :return: returns the hausdorff distance between the target and the prediction masks without the exclusion area.
    '''

    # TODO: Check if the masks are binary and their dimensions
    # TODO: Check if the edges of the prediction after the exclusion are still values of 1 and 0.


    erosion_kernel = ndimage.generate_binary_structure(2, 1)

    target_erosion = ndimage.binary_erosion(target, erosion_kernel)
    target_edge = (target > 0) ^ target_erosion
    prediction_erosion = ndimage.binary_erosion(prediction, erosion_kernel)
    prediction_edge = (prediction > 0) ^ prediction_erosion
    exclusion_erosion = ndimage.binary_erosion(exclusion, erosion_kernel)
    exclusion_edge = (exclusion > 0) ^ exclusion_erosion

    # Subtract the exclusion edge from the prediction edge
    prediction_edge = prediction_edge - exclusion_edge

    c1 = np.array(np.where(target_edge == 1)).transpose()
    c2 = np.array(np.where(prediction_edge == 1)).transpose()

    d = cdist(c1, c2)
    d1 = np.amin(d, axis=0)
    d2 = np.amin(d, axis=1)

    hausdorff = max(max(d1), max(d2))
    mad = sum([sum(d1) / len(d1), sum(d2) / len(d2)]) / 2

    return hausdorff, mad


def calcMSE(img1, img2):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def calcSSIM(img1, img2):
    return ssim(img1, img2)


def post_process_prediction(pred):
    # Post-processes the prediction array by getting the connected components and turning it into a boolean mask
    pred = np.round(pred / np.max(pred)).astype("int8")

    # Remove small connected components in every slice of the prediction
    for i, p in enumerate(pred):
        selem = morphology.star(int(np.max(np.shape(p)) / 300))
        p = morphology.binary_closing(p, selem)
        p = morphology.binary_opening(p, selem)
        pred[i, :, :] = p

    # Turns the prediction into a boolean mask based on the 0.5 threshold
    pred = pred > 0


    return pred


def evaluate_rtstruct_structures(rtstruct_path, patient_path,
                           structure1='PTV_tot', structure2='Revision (final) - PTV_tot_pred', metrics=None):
    '''
    This function compares the same rtstruct on two different structures within the same patient.
    :param rtstruct_path:
    :param patient_path: path where the DICOM images of that patient are stored
    :param structure1: name of the first structure to compare
    :param structure2: name of the second structure to compare (ground truth)
    :return:
    '''

    if metrics is None:
        metrics = ['dice', 'hd', 'hd95']

    patient_name = patient_path.split('/')[-1]

    # Load the RTstructs
    rtstruct = RTStructBuilder.create_from(dicom_series_path=patient_path, rt_struct_path=rtstruct_path)


    roi3dMask = rtstruct.get_roi_mask_by_name(structure1)
    roi3dMask = np.round(roi3dMask / np.max(roi3dMask)).astype("int8")

    roi3dMask2 = rtstruct.get_roi_mask_by_name(structure2)
    roi3dMask2 = np.round(roi3dMask2 / np.max(roi3dMask2)).astype("int8")

    listImages = os.listdir(patient_path)
    firstImagePath = os.path.join(patient_path, listImages[0])
    ds = pydicom.dcmread(firstImagePath)

    st = ds.SliceThickness
    ps = ds.PixelSpacing
    spacing = [st, ps[0], ps[1]]

    if roi3dMask.shape == roi3dMask2.shape:
        res = sg.write_metrics(labels=[1], gdth_img=roi3dMask2, pred_img=roi3dMask, csv_file=None,  spacing=spacing, metrics=metrics)
    else:
        print(f'Shape mismatch. pred={str(roi3dMask.shape)} and gt={str(roi3dMask2.shape)}')
        res = None

    if res is not None:
        patientResults = {'patient': patient_name, 'pred': structure1, 'target': structure2,
                          'dice': res[0]['dice'][0], 'hd': res[0]['hd'][0], 'hd95': res[0]['hd95'][0]}
        print(res[0]['dice'][0], res[0]['hd'][0], res[0]['hd95'][0])
        return patientResults
    else:
        return None


def evaluate_folder_npz(folder_npz_path, folder_dicom_path, evaluation_results_path, structure, metrics=None):
    results = pd.DataFrame(columns=['patient', 'pred', 'target', 'dice', 'hd', 'hd95'])

    for i, patientFile in enumerate(os.listdir(folder_npz_path)):
        print(patientFile)
        patient_path = os.path.join(folder_npz_path, patientFile)
        patient_name = patientFile.split('.')[0]

        patient_dicom_path = os.path.join(folder_dicom_path, 'imgs', patient_name)
        dicom_image_list = os.listdir(patient_dicom_path)
        first_dicom_image_path = os.path.join(patient_dicom_path, dicom_image_list[0])
        ds = pydicom.dcmread(first_dicom_image_path)

        st = ds.SliceThickness
        ps = ds.PixelSpacing
        spacing = [st, ps[0], ps[1]]
        res = binary_compare(patient_path, spacing, metrics=metrics)

        if res is not None:
            patientResults = {'patient': patient_name , 'pred': structure, 'target': structure,
                              'dice': res[0]['dice'][0], 'hd': res[0]['hd'][0], 'hd95': res[0]['hd95'][0]}
            print(res[0]['dice'][0], res[0]['hd'][0], res[0]['hd95'][0])
            results = pd.concat([results, pd.DataFrame(patientResults, index=[i])], ignore_index=True)

    results.to_excel(evaluation_results_path)