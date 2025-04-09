import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pydicom


def compressImageTo8bit(imagePath):
    '''
    This function compresses linearly the image to 8 bits.
    :param imagePath:
    :return:
    '''
    image = plt.imread(imagePath)
    return (image / np.max(image) * 255).astype('uint8')


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



def calculateMSE(image1, image2):
    return np.mean((image1 - image2) ** 2)


def nonLinearLUT(imageHU):
    '''
    This function uses a non-linear LUT to transform the image into a more useful HU range and compresses it into 8-bits
     for the clinician interpretation.
    :param imageHU:
    :return:
    '''
    imageHUCopy = imageHU.copy()
    # If the pixel is between -1000 and -300 HU, turn it into a scale between 0 and 20
    for x in range(imageHUCopy.shape[0]):
        for y in range(imageHUCopy.shape[1]):

            # If the intensity is between -1000 and -300 HU, turn it into a scale between 0 and 20
            if -1000 <= imageHUCopy[x, y] <= -300:
                imageHUCopy[x, y] = (imageHUCopy[x, y] + 1000) / 35
            elif -300 < imageHUCopy[x, y] <= -150:
                imageHUCopy[x, y] = (imageHUCopy[x, y] + 600) / 15
            elif -150 < imageHUCopy[x, y] <= -50:
                imageHUCopy[x, y] = (imageHUCopy[x, y] + 300) / 5
            elif -50 < imageHUCopy[x, y] <= 0:
                imageHUCopy[x, y] = (imageHUCopy[x, y] + 175) / 2.5
            elif 0 < imageHUCopy[x, y] <= 100:
                imageHUCopy[x, y] = (imageHUCopy[x, y] + 140) / 2
            elif 100 < imageHUCopy[x, y] <= 200:
                imageHUCopy[x, y] = (imageHUCopy[x, y] + 200) / 2.5
            elif 200 < imageHUCopy[x, y] <= 400:
                imageHUCopy[x, y] = (imageHUCopy[x, y] + 600) / 5
            elif 400 < imageHUCopy[x, y] <= 800:
                imageHUCopy[x, y] = (imageHUCopy[x, y] + 2800) / 16
            elif 800 < imageHUCopy[x, y] <= 1650:
                imageHUCopy[x, y] = (imageHUCopy[x, y] + 10450) / 50
            elif 1650 < imageHUCopy[x, y] <= 3050:
                imageHUCopy[x, y] = (imageHUCopy[x, y] + 22550) / 100

    compressedImage = imageHUCopy.astype('uint8')
    return compressedImage


def inverseNonLinearLUT(image):
    imageHU = image.copy()

    imageDecomp = np.zeros(imageHU.shape, dtype='uint16')

    for x in range(imageHU.shape[0]):
        for y in range(imageHU.shape[1]):
            if 0 <= imageHU[x, y] <= 20:
                imageDecomp[x, y] = imageHU[x, y] * 35 - 1000 + 1000
            elif 21 <= imageHU[x, y] <= 30:
                imageDecomp[x, y] = imageHU[x, y] * 15 - 600 + 1000
            elif 31 <= imageHU[x, y] <= 50:
                imageDecomp[x, y] = imageHU[x, y] * 5 - 300 + 1000
            elif 51 <= imageHU[x, y] <= 70:
                imageDecomp[x, y] = imageHU[x, y] * 2.5 - 175 + 1000
            elif 71 <= imageHU[x, y] <= 120:
                imageDecomp[x, y] = imageHU[x, y] * 2 - 140 + 1000
            elif 121 <= imageHU[x, y] <= 160:
                imageDecomp[x, y] = imageHU[x, y] * 2.5 - 200 + 1000
            elif 161 <= imageHU[x, y] <= 200:
                imageDecomp[x, y] = imageHU[x, y] * 5 - 600 + 1000
            elif 201 <= imageHU[x, y] <= 225:
                imageDecomp[x, y] = imageHU[x, y] * 16 - 2800 + 1000
            elif 226 <= imageHU[x, y] <= 242:
                imageDecomp[x, y] = imageHU[x, y] * 50 - 10450 + 1000
            elif 243 <= imageHU[x, y] <= 255:
                imageDecomp[x, y] = imageHU[x, y] * 100 - 22550 + 1000

    return imageDecomp


def decompress12bits(image):
    '''
    This function decompresses an 8-bit image into a 12-bit image and stores it as a 16-bit image.
    :param image:
    :return:
    '''
    return (image/255 * 4095).astype('uint16')


def getMask(automi_path, patient, structure):
    '''
    This function gets the mask for a given patient and structure.
    :param patient:
    :param structure:
    :return:
    '''

    try:
        maskPath = os.path.join(automi_path, 'masks', patient, structure + '.npz')
        arr = np.load(maskPath)
        mask = arr['arr_0']
    except:
        mask = False

    return mask


def calculateMSE_ROI(image1, image2, maskSlice):
    '''
    This function calculates the MSE between two images, but only in the ROI.
    :param image1:
    :param image2:
    :param roiImage:
    :return:
    '''

    # Calculate the MSE only in the ROI
    roiImage = maskSlice.astype('bool')
    mse = np.mean((image1[roiImage] - image2[roiImage]) ** 2)

    return mse

def compressDecompress(automi_path, patientList, resultsPath):
    for patient in patientList:

        patientPath = os.path.join(automi_path, 'imgs', patient)
        patientResultsPath = os.path.join(resultsPath, patient)
        if not os.path.isdir(patientResultsPath):
            os.mkdir(patientResultsPath)

        patientDecompressionMSE = []
        #mask = getMask(automi_path, patient, 'BODY')

        for i, imageFilename in enumerate(os.listdir(patientPath)):
            # Read the image
            imagePath = os.path.join(patientPath, imageFilename)
            imageName = imageFilename.split('.')[0]
            image = pydicom.dcmread(imagePath)

            # Compress and decompress the image
            imageHU = imageToHU(image)
            compressedImage = nonLinearLUT(imageHU)
            #decompressedImage = cv2.normalize(compressedImage, None, 0, 4095, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
            decompressedImage = inverseNonLinearLUT(compressedImage)

            # Calculate the MSE between the compressed image and the decompressed image.
            # if mask is not False:
            #     patientDecompressionMSE.append(calculateMSE_ROI(image.pixel_array, decompressedImage, mask[:, :, i]))
            # else:
            patientDecompressionMSE.append(calculateMSE(image.pixel_array, decompressedImage))

            # Saving the images in the results' path.
            originalName = f'{imageName}_original.png'
            compressedName = f'{imageName}_compressed.png'
            decompressedName = f'{imageName}_decompressed.png'
            newDicomName = f'{imageName}_decompressed.dcm'
            result_ori_path = os.path.join(patientResultsPath, originalName)
            result_com_path = os.path.join(patientResultsPath, compressedName)
            result_decom_path = os.path.join(patientResultsPath, decompressedName)
            result_nDicom_path = os.path.join(patientResultsPath, newDicomName)

            # plt.imsave(result_ori_path, imageHU, cmap='gray')
            # plt.imsave(result_com_path, compressedImage, cmap='gray')
            # plt.imsave(result_decom_path, decompressedImage, cmap='gray')

            # Altering the original DICOM image and saving it in the results' path.
            image.PixelData = decompressedImage.tobytes()
            image.SOPInstanceUID = pydicom.uid.generate_uid()
            image.save_as(result_nDicom_path, write_like_original=False)

        # Calculating the mean MSE for the patient.
        mse_mean = np.mean(patientDecompressionMSE)
        print(f'Mean MSE for patient {patient}: {mse_mean}')



if __name__ == "__main__":
   automi_path = sys.argv[1]
   resultsPath = sys.argv[2]

   imgs_path = os.path.join(automi_path, 'imgs')
   patientList = os.listdir(imgs_path)
   # First 20 elements of patientList
   patientList = patientList[:25]
   
   compressDecompress(automi_path, patientList, resultsPath)
