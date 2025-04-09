'''
Turns a nifti segmentation into a newly created DICOM RTSTRUCT file.
'''

from rt_utils import RTStructBuilder
import nibabel as nib
import numpy as np

rtstruct = RTStructBuilder.create_from(dicom_series_path='/mnt/storage/ricardo/AUTOMI/AUTOMI_new_data/dicom/05082024',

# Read nifti segmentation prediction file
nifti_pred = nib.load('/mnt/storage/ricardo/AUTOMI/AUTOMI_new_data/pred/05082024.nii.gz')
nifti_pred = nifti_pred.get_fdata()
nifti_pred = nifti_pred.transpose(1, 0, 2)
# flip the nifti_pred to match the DICOM coordinate system
nifti_pred = np.flip(nifti_pred, axis=0)
nifti_pred = np.round(nifti_pred / np.max(nifti_pred)).astype("int8")
nifti_pred = nifti_pred > 0

rtstruct.add_roi(mask=nifti_pred, name='CTV_pred', roi_number=34)

rtstruct.save('/mnt/storage/ricardo/AUTOMI/AUTOMI_new_data/rtstruct/05082024/05082024.dcm')