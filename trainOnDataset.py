import datapaths
import utils_data
import utils_train
import os
import torch
import csv
from torchsummary import summary
from utils_model import get_model, save_model_params, load_model_params
import sys


def train(modelParams, resultsPath):
    trainPaths = modelParams['trainPaths']

    # Results paths
    if not os.path.exists(resultsPath):
        os.mkdir(resultsPath)
    path_results = os.path.join(resultsPath, modelParams['model_folder'])
    if not os.path.exists(path_results):
        os.mkdir(path_results)

    utils_train.save_model_params(resultsPath, modelParams)

    isCudaAvailable = torch.cuda.is_available()
    print(f'Training on GPU: {isCudaAvailable}')

    # Number of gpus
    multiGPU = False
    if isCudaAvailable:
        gpuCount = torch.cuda.device_count()
        print(f'{gpuCount} GPU\'s detected.')
        if gpuCount > 1:
            multiGPU = True

    trainTransforms = utils_data.TrainTransforms(modelParams,
                                                 toTensor=True)

    data = utils_data.DatasetFolderAutomi(datasetPath=trainPaths,
                                          structure='PTV_tot',
                                          transform=trainTransforms)

    print('Found {} samples for training.'.format(len(data.allImagesPaths)))

    foldList = modelParams['fold_list']
    for foldNumber in foldList:
        modelFilename = os.path.join(path_results, f'model_{foldNumber}.pt')

        # Path of the csv that contains the distribution for training and validation.
        foldDistCSVPath = os.path.join(trainPaths, f'imageDist_fold{foldNumber}.csv')

        # Gets the indexes of the images used for training and the indexes used for validation.
        trainIndexes = utils_data.getIndexes(foldDistCSVPath, subset='train')
        valIndexes = utils_data.getIndexes(foldDistCSVPath, subset='val')

        # Creates the torch sampler using the training and the validation indexes.
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(trainIndexes)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(valIndexes)

        # Shows the number of images used for training and the number of images used for validaton.
        print('Training set: {}'.format(len(trainIndexes)))
        print('Validation set: {}'.format(len(valIndexes)))

        # Dataloader iterators
        dataloaders = {
            'train': torch.utils.data.DataLoader(data, batch_size=modelParams['batch_size'], sampler=train_sampler,
                                                 shuffle=False, num_workers=4, pin_memory=True),
            'val': torch.utils.data.DataLoader(data, batch_size=modelParams['batch_size'], sampler=val_sampler,
                                                 shuffle=False, num_workers=4, pin_memory=True)}

        model = get_model(modelParams)

        if multiGPU:
            summary(model.module, torch.rand((1, 1, modelParams['size_input'], modelParams['size_input'])))
        else:
            summary(model, torch.rand((1, 1, modelParams['size_input'], modelParams['size_input'])))

        # Depending on the experiment, singleOutput is False or True
        lossFunction = utils_train.GenericLoss(dataloaders['train'],
                                           weight_func=[getattr(utils_train, lw) for lw in modelParams['lossweights']],
                                           criterion=[getattr(utils_train, l) for l in modelParams['loss']],
                                           singleOutput=True)

        accFunction = utils_train.GenericAcc([utils_train.ClassAcc])

        valpred_fpath = os.path.join(path_results, 'val_examples')
        if not os.path.isdir(valpred_fpath):
            os.mkdir(valpred_fpath)
        valPredictions = utils_train.dataPredictions(fpath=valpred_fpath)

        n_batches = int(modelParams['n_imgs_per_epochs'] / modelParams['batch_size'])
        model, history = utils_train.train(model, dataloaders['train'], dataloaders['val'],
                                           modelFilename, lossFunction, accFunction,
                                           valPredictions=valPredictions,
                                           max_epochs_stop=modelParams['max_epochs_stop'],
                                           n_epochs=modelParams['n_epochs'],
                                           n_batches=n_batches,
                                           learning_rate=modelParams['learning_rate'])

        with open(os.path.join(path_results, f'fold{foldNumber}history.csv'), 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
            filewriter.writerows(history)


if __name__ == '__main__':

    # If bash arguments are present
    if len(sys.argv) > 1:
        path_train = sys.argv[1]
        path_results = sys.argv[2]
    else:
        path_train = datapaths.datapaths['AUTOMI']
        path_results = datapaths.resultspath

    modelParams = {'model_folder': 'AUTOMI_PTV_tot_BCE',
                   'model_name': 'unet',
                   'freeze': False,
                   'fold_list': [2,3,4,5],
                   'batch_size': 4,
                   'size_input': 512,
                   'loss': ['BCELoss'],
                   'lossweights': ['getLossWeights_None'],
                   'max_epochs_stop': 20,
                   'n_epochs': 100,
                   'n_imgs_per_epochs': 25000,
                   'learning_rate': 1e-5,
                   'trainPaths': path_train}
    # Results paths
    train(modelParams, path_results)
