import os
import numpy as np
import datapaths

# Read the first dicom image of each patient and calculate the number of slices in the patient
sliceCounterPatients = []
imagesPath = os.path.join(datapaths.datapaths['AUTOMI'], 'imgs')

for patient in os.listdir(imagesPath):
    patientPath = os.path.join(imagesPath, patient)
    sliceCounterPatients.append(len(os.listdir(patientPath)))

print((sliceCounterPatients))
print(max(sliceCounterPatients))
print(min(sliceCounterPatients))
print(np.mean(sliceCounterPatients))
