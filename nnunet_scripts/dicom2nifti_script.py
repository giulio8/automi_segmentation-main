import dicom2nifti


if __name__ == '__main__':
    dicom2nifti.convert_directory(
        '/mnt/storage/ricardo/AUTOMI/AUTOMI_new_data/dicom/05082024/',
        '/mnt/storage/ricardo/AUTOMI/AUTOMI_new_data/nifti/',
        compression=True,
        reorient=True
    )
