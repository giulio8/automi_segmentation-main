
import os, sys
sys.path.insert(0, '..')
import datapaths
from utils_model import get_model, save_model_params, load_model_params
from utils_data import TestTransform
import utils_data
from utils_train import dataPredictions
from torch import no_grad
from torch import load as torchload
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import time
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def main(model_folder, datasetlist, structure, img_path, dispFlag=False, subsetlist=[''], nfolds=5):
    nmodels = 0
    for foldNumber in range(1, nfolds + 1):
        if os.path.isfile(os.path.join(model_folder, f'model_{foldNumber}.pt')):
            nmodels += 1
        else:
            break

    # Load model params
    model_params = load_model_params(model_folder)

    # Image transformations
    image_transform = TestTransform(model_params)

    if nmodels > len(datasetlist):
        for datastr, subset in zip(datasetlist, subsetlist):
            dataset, path_results = loadData(datastr, subset, structure, image_transform, model_folder, img_path)
            for foldNumber in range(1, nmodels + 1):
                if os.path.isfile(os.path.join(path_results, f'fold{foldNumber}predictions.csv')):
                    print('Predictions already made for fold {} - {}.'.format(foldNumber, datastr))
                    continue

                if not os.path.isfile(os.path.join(model_folder, f'fold{foldNumber}history.csv')):
                    print(f'Model fold {foldNumber} has not finished training.')
                    continue

                model = loadModel(model_folder, foldNumber, model_params)
                dataloader = getDataLoader(dataset, datastr, subset, foldNumber, img_path)
                predict(model, dataloader, path_results, foldNumber, dispFlag=dispFlag)
    else:
        for foldNumber in range(1, nmodels + 1):
            if not os.path.isfile(os.path.join(model_folder, 'fold{}history.csv'.format(foldNumber))):
                print('Model fold {} has not finished training.'.format(foldNumber))
                continue

            model = loadModel(model_folder, foldNumber, model_params)
            for datastr, subset in zip(datasetlist, subsetlist):
                dataset, path_results = loadData(datastr, subset, structure, image_transform, model_folder, img_path)
                if os.path.isfile(os.path.join(path_results, 'fold{}predictions.csv'.format(foldNumber))):
                    print('Predictions already made for fold {} - {}.'.format(foldNumber, datastr))
                    continue
                dataloader = getDataLoader(dataset, datastr, subset, foldNumber, img_path)
                predict(model, dataloader, path_results, foldNumber, dispFlag=dispFlag)


def loadData(dataset_name, subset, structure, image_transform, model_folder, img_path):
    '''
    Load data and create path to save results
    :param dataset_name: is the name of the dataset
    :param subset:
    :param structure:
    :param image_transform:
    :param model_folder:
    :return:
    '''
    # Path to subset results
    if subset == '':
        path_results = os.path.join(model_folder, dataset_name)
    else:
        path_results = os.path.join(model_folder, '{}_{}'.format(dataset_name, subset))

    if not os.path.isdir(path_results):
        os.mkdir(path_results)

    # Datasets from each folder
    data = utils_data.DatasetFolderAutomi(datasetPath=img_path,
                                          structure=structure,
                                          transform=image_transform,
                                          datasetCSV=None)
    print('Found {} samples.'.format(len(data.allImagesPaths)))

    return data, path_results


def loadModel(model_folder, foldNumber, model_params):

    model_path = os.path.join(model_folder, 'model_{}.pt'.format(foldNumber))
    print('Loading', model_path)

    # Load model
    model = get_model(model_params)
    model_dict = torchload(model_path)
    model_dict = {k: model_dict[k] for k in model.state_dict()}
    model.load_state_dict(model_dict)

    return model


def getDataLoader(dataset, dataset_name, subset, foldNumber, dataset_path):

    # Path of the csv that contains the distribution for training and validation.
    foldDistCSVPath = os.path.join(dataset_path, f'imageDist_fold{foldNumber}.csv')

    testIndexes = utils_data.getIndexes(foldDistCSVPath, subset=subset)
    testSampler = SubsetRandomSampler(testIndexes)
    print('{} set: {}'.format(dataset_name, len(testIndexes)))

    # Dataloader iterators
    dataloader = DataLoader(dataset, batch_size=10, sampler=testSampler, shuffle=False)

    return dataloader


def predict(model, dataloader, path_results, foldNumber, dispFlag=False):
    path_results_k = os.path.join(path_results, f'fold{foldNumber}predictions')
    if not os.path.isdir(os.path.join(path_results_k)):
        os.mkdir(path_results_k)

    dPredictions = dataPredictions()
    with no_grad():
        model.eval()
        nbatches = len(dataloader)
        st_time = time.time()
        for ii, (data, target, info) in enumerate(dataloader):
            output = model(data)
            if dispFlag:
                for d, o in zip(data, output):
                    image = d.cpu().permute(1, 2, 0).detach().numpy()
                    segm = o.cpu().permute(1, 2, 0).detach().numpy()
                    fig, (ax0, ax1) = plt.subplots(1, 2)
                    ax0.imshow(image)
                    ax1.imshow(segm)
                    plt.show()

            dPredictions.append(data, output, target=target, info=info)
            print(f'\rFold {foldNumber}: \t{100 * (ii + 1) / nbatches:.2f}% complete. {time.time() - st_time:.2f} seconds elapsed in fold.', end='')
            dPredictions.write(path_results_k, clear=True)

    print(f'Fold {foldNumber}: \t{100 * (ii + 1) / nbatches:.2f}% complete. {time.time() - st_time:.2f} seconds elapsed in fold.')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_folder = sys.argv[1]
        img_path = sys.argv[2]
    else:
        model_folder = os.path.join(datapaths.resultspath, 'AUTOMI_PTV_tot_BCE_Loss_May_2023')
        img_path = datapaths.datapaths['AUTOMI']

    datasetlist = ['AUTOMI']
    subsetlist = ['test']
    structure = 'PTV_tot'

    main(model_folder, datasetlist, structure, img_path, dispFlag=False, subsetlist=subsetlist)