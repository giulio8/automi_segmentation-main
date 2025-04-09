import datapaths
import os
import nibabel as nib
import numpy as np
import pydicom

def save_nifti(data, result_path, affine):
    """
    Saves a 3D numpy array as a NIfTI file.
    Args:
        filename (str): The output file path.
        data (numpy.ndarray): A 3D numpy array.
    """
    if len(data.shape) != 3:
        raise ValueError('Data must be 3D.')

    nii = nib.Nifti1Image(data, affine)  # create NIfTI object
    nib.save(nii, result_path)  # save to file

def calculate_affine_matrix(ds):
    # Get pixel spacing
    pixel_spacing = [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])]

    # Get the orientation of the image
    image_orientation = [float(i) for i in ds.ImageOrientationPatient]

    # Get the position of the image
    image_position = [float(i) for i in ds.ImagePositionPatient]

    # Create the affine matrix
    affine_matrix = np.array([[image_orientation[0]*pixel_spacing[0], image_orientation[3]*pixel_spacing[1], 0, image_position[0]],
                              [image_orientation[1]*pixel_spacing[0], image_orientation[4]*pixel_spacing[1], 0, image_position[1]],
                              [image_orientation[2]*pixel_spacing[0], image_orientation[5]*pixel_spacing[1], 0, image_position[2]],
                              [0, 0, 0, 1]])
    return affine_matrix

def calculate_affine_matrix2(ds):
    print(ds.SliceThickness)
    affine_matrix = np.diag([ds.PixelSpacing[0], ds.PixelSpacing[1], ds.SliceThickness, 1])
    affine_matrix[:3, 3] = ds.ImagePositionPatient
    return affine_matrix

def calculate_affine_matrix_3d(ds):
    ...

def main():
    # patient = '97b1d50066'
    #
    # patient_path = os.path.join(datapaths.datapaths['AUTOMI'], 'imgs', patient)
    # first_dcm_file_path = os.listdir(patient_path)[0]
    # ds = pydicom.dcmread(os.path.join(patient_path, first_dcm_file_path))
    # affine_matrix = calculate_affine_matrix(ds)
    # affine_matrix2 = calculate_affine_matrix2(ds)
    #
    # print(affine_matrix)
    # print(affine_matrix2)
    #
    # npz_file_path = os.path.join(datapaths.resultspath, 'AUTOMI_PTV_tot_BCELoss', 'AUTOMI_test','fold4predictionsNPZ', f'{patient}.npz')
    # npz_file = np.load(npz_file_path)
    # pred = npz_file['pred']
    #
    # pred = np.round(pred / np.max(pred)).astype("int8")
    # pred = (pred > 0.5).astype(np.uint8)
    #
    # #pred = pred.transpose(2, 1, 0)
    # save_nifti(pred, f'{patient}.nii')  # save to file

    nifti_file_path = ''
    npz_file_path = ''

    nifti = nib.load(nifti_file_path)
    npz_file = np.load(npz_file_path)

    # Get the affine data from the nifti
    affine = nifti.affine

    # Turn the npz into nifti using the affine data
    npz = npz_file[0]

    save_nifti(npz, 'result.nii', affine)


if __name__ == '__main__':
    main()
