import numpy as np
import os, sys
import datapaths
import pandas as pd
import pydicom
import utils_data
from matplotlib import pyplot as plt

def setDicomWinWidthWinCenter(img_data, winwidth, wincenter):
    img_temp = img_data
    min = (2 * wincenter - winwidth) / 2.0 + 0.5
    max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    img_temp = (img_temp - min) * dFactor

    img_temp[img_temp < 0] = 0
    img_temp[img_temp > 255] = 255
    return img_temp


def imageToHU(image):

    intercept = image.RescaleIntercept
    slope = image.RescaleSlope
    imageHU = image.pixel_array * slope + intercept

    return imageHU


def normalize_image(image, center, width):
    # image = transform_to_hu result

    img_min = center - width // 2
    img_max = center + width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max

    window_image = (255 * (window_image - img_min)) // width

    return window_image

def normalizeImage(image, max):
    return image/max

if __name__ == "__main__":

    patientPath = os.path.join(datapaths.datapaths['AUTOMI'], 'imgs', '1b20a3cb66')
    resultsPath = os.path.join(datapaths.datapaths['AUTOMI'], 'results', '1b20a3cb66')


    for i, image in enumerate(os.listdir(patientPath)):

        imagePath = os.path.join(patientPath, image)
        imageDicom = pydicom.dcmread(imagePath)
        image = imageDicom.pixel_array
        imageHU = imageToHU(imageDicom)

        plt.subplot(2, 2, 1)
        title = 'Simple 8-bit conversion'
        plt.title(title, size=7)
        imageSimpleConversion = image.astype(np.int8)
        plt.imshow(imageSimpleConversion, cmap='Greys_r')


        plt.subplot(2, 2, 2)
        title = 'Normalized + 8-bit conv'
        plt.title(title, size=7)
        imageNormalized = normalizeImage(image, 4095) * 255
        imageNormalized = imageNormalized.astype(np.int8)
        plt.imshow(imageNormalized, cmap='Greys_r')

        plt.subplot(2, 2, 3)
        title = 'Norm + lung LUT'
        plt.title(title, size=7)
        imageNormalizedLungs = normalize_image(imageHU, -500, 1800)
        plt.imshow(imageNormalizedLungs, cmap='Greys_r')

        plt.subplot(2, 2, 4)
        title = 'Norm + heart LUT'
        plt.title(title, size=7)
        imageNormalizedHeart = normalize_image(imageHU, 30, 350)
        plt.imshow(imageNormalizedHeart, cmap='Greys_r')
        filePath = os.path.join(resultsPath, 'b51022200a_{}'.format(i))
        plt.savefig(filePath)
        plt.show()



'''
"CTwindow_level": {
    "coarse": -500,
    "LeftLung": -500,
    "RightLung": -500,
    "Heart": 30,
    "Esophagus": 85,
    "Trachea": -440,
    "SpinalCord": 0
  },
  "CTwindow_width": {
    "coarse": 1800,
    "LeftLung": 1800,
    "RightLung": 1800,
    "Heart": 350,
    "Esophagus": 324,
    "Trachea": 1180,
    "SpinalCord": 600
  },
'''
