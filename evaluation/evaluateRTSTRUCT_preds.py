import utils_evaluation

rtstruct_path = '/home/ricardo/Desktop/ricardo/results/CityOfHope/rt_structs/Linac2/Linac2.dcm'
structure1 = 'PTV_tot_pred'
structure2 = 'Revision (final) - PTV_tot_pred'
patient_path = '/home/ricardo/Desktop/ricardo/CityOfHope/Linac2/'

patient_results = utils_evaluation.evaluate_rtstruct_structures(rtstruct_path, patient_path, structure1, structure2)
print(patient_results)
