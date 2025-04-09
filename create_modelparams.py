import pickle
import os

modelParams = {
    'model_folder': 'AUTOMI_PTV_tot_BCE',
    'model_name': 'unet',
    'freeze': False,
    'fold_list': [2, 3, 4, 5],
    'batch_size': 4,
    'size_input': 512,
    'loss': ['BCELoss'],
    'lossweights': ['getLossWeights_None'],
    'max_epochs_stop': 20,
    'n_epochs': 100,
    'n_imgs_per_epochs': 25000,
    'learning_rate': 1e-5,
    'trainPaths': '/automi_seg/data'
}

results_dir = '/automi_seg/results/'
model_folder = 'giulio'
os.makedirs(os.path.join(results_dir, model_folder), exist_ok=True)

with open(os.path.join(results_dir, model_folder, 'modelparams.pkl'), 'wb') as f:
    pickle.dump(modelParams, f)

print('✅ modelparams.pkl salvato in', os.path.join(results_dir, model_folder))
