import numpy as np
from evaluateOnDatasetNPZ import binary_compare

evaluate_npz_path = 'C:/Users/Ricardo/Desktop/AUTOMI_00043.npz'

# Load the npz file
npz_file = np.load(evaluate_npz_path)
pred = npz_file['pred']
target = npz_file['target']

print(pred.min())
print(target.min())

