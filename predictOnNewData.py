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
import time

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def predict_single_volume(model, image_tensor, model_params):
    """
    Performs a forward pass on a single image tensor (already preprocessed).
    Returns the raw output of the model without writing to disk.
    :param model: the loaded PyTorch model
    :param image_tensor: a single image tensor of shape (1, C, H, W, D) or (C, H, W, D)
    :param model_params: dictionary of model parameters (in case normalization is needed)
    :return: model prediction tensor
    """
    from torch import no_grad

    if image_tensor.ndim == 4:
        image_tensor = image_tensor.unsqueeze(0)  # add batch dimension

    image_tensor = image_tensor.float().cuda()  # send to GPU if needed

    with no_grad():
        model.eval()
        output = model(image_tensor)

    return output

def main(model_folder, img_path, path_results, foldNumber, structure=None):
    '''
    This function is used to predict on new data, it uses a model trained on the AUTOMI dataset from a specific fold.
    If the new dataset has masks for a specific structure, the structure is saved as an image as well.
    :param img_path:
    :param model_folder: is the path to the folder where the model is saved
    :param path_results: is the path to the folder where the results will be saved
    :param foldNumber: is the fold number of the model to be used
    :param structure: is the structure groundtruth that will be saved as an image along the prediction and the original image.
    if there is no groundtruth for the structure, it should be None
    :return:
    '''
    # Load model params
    model_params = load_model_params(model_folder)

    # Image transformations
    image_transform = TestTransform(model_params)

    # Load the data for the custom dataset
    data = utils_data.DatasetFolderAutomi(datasetPath=img_path,
                                        structure=structure,
                                        transform=image_transform,
                                        datasetCSV=None)
    print('Found {} samples.'.format(len(data.allImagesPaths)))

    model = loadModel(model_folder, foldNumber, model_params)
    dataloader = DataLoader(data, batch_size=10, sampler=None, shuffle=False)

    predict(model, dataloader, path_results, foldNumber)



def loadModel(model_folder, foldNumber, model_params):

    model_path = os.path.join(model_folder, 'model_{}.pt'.format(foldNumber))
    print('Loading', model_path)

    # Load model
    model = get_model(model_params)
    model_dict = torchload(model_path)
    model_dict = {k: model_dict[k] for k in model.state_dict()}
    model.load_state_dict(model_dict)

    return model


def predict(model, dataloader, path_results, foldNumber):
    path_results_k = os.path.join(path_results, f'fold{foldNumber}predictions')
    if not os.path.isdir(os.path.join(path_results_k)):
        os.mkdir(path_results_k)
    preds = dataPredictions()

    with no_grad():
        model.eval()
        nbatches = len(dataloader)
        st_time = time.time()
        for ii, (data, target, info) in enumerate(dataloader):
            output = model(data)

            preds.append(data, output, target=target, info=info)
            print(
                f'\rNew Dataset: \t{100 * (ii + 1) / nbatches:.2f}% complete. {time.time() - st_time:.2f} seconds elapsed.',
                end='')
            preds.write(path_results_k, clear=True)

    print(
        f'New Dataset: \t{100 * (ii + 1) / nbatches:.2f}% complete. {time.time() - st_time:.2f} seconds elapsed.')
    



if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_folder = sys.argv[1]
        img_path = sys.argv[2]
        path_results = sys.argv[3]
    else:
        model_folder = os.path.join(datapaths.resultspath, 'AUTOMI_PTV_tot_BCE_Loss_May_2023')
        img_path = datapaths.datapaths['CityOfHope']
        path_results = os.path.join(datapaths.resultspath, 'CityOfHope')

    fold_number = 5
    main(model_folder, img_path, path_results, fold_number)