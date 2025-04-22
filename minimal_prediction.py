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

def predict_single_volume(model, image_tensor):
    """
    Performs a forward pass on a single image tensor (already preprocessed).
    Returns the raw output of the model without writing to disk.
    :param model: the loaded PyTorch model
    :param image_tensor: a single image tensor of shape (1, C, H, W, D) or (C, H, W, D)
    :param model_params: dictionary of model parameters (in case normalization is needed)
    :return: model prediction tensor
    """

    if image_tensor.ndim == 4:
        image_tensor = image_tensor.unsqueeze(0)  # add batch dimension

    image_tensor = image_tensor.float().cuda()  # send to GPU if needed

    with no_grad():
        model.eval()
        output = model(image_tensor)

    return output

def predict(model, data_loader):
    """
    Predicts on a dataset using the provided model.
    :param model: the loaded PyTorch model
    :param data_loader: DataLoader object for the dataset
    :return: predictions as a list of tensors
    """
    
    predictions = []
    
    for batch in data_loader:
        image_tensor = batch['image'].cuda()  # send to GPU if needed
        pred = predict_single_volume(model, image_tensor)
        predictions.append(pred.cpu())  # move back to CPU if needed

    return predictions

def loadModel(model_path):

    print('Loading', model_path)

    model = torchload(model_path)

    return model