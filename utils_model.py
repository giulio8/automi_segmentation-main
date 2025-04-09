try:
    from torchvision import models
    import torch.nn as nn
    import torch
    import torch.nn.functional as F
except:
    print('Could not load PyTorch... Continuing anyway!')
    
import os
import numpy as np
import pickle
import copy

from unet import UNet
    
def get_model(model_params):
    model = get_pretrained_model(model_params)
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model,device_ids=[0,1])
    else:
        model = nn.DataParallel(model)

    model.to('cuda')   
    
    return model

def load_model_params(fpath):
    model_params = pickle.load(open( os.path.join(fpath,'modelparams.pkl'), "rb"))
    return model_params
    
def save_model_params(fpath,model_params):
    with open(os.path.join(fpath,'modelparams.pkl'), 'wb') as f:
        pickle.dump(model_params, f)

def get_pretrained_model(model_params):
    if model_params['model_name'] == 'unet':
        unet_params = {'in_channels': 1, 'out_channels': 1 , 'n_blocks': 5,
                       'start_filters': 32, 'activation': 'relu', 'normalization': 'batch',
                       'conv_mode': 'same', 'dim': 2, 'up_mode': 'transposed'}
        for k,_ in unet_params.items():
            if k in model_params:
                unet_params[k] = model_params[k]

        model = UNet(**unet_params)
    elif model_params['model_name'] == 'unet_multiclass':
        unet_params = {'in_channels': 1, 'out_channels': 3, 'n_blocks': 5,
                       'start_filters': 32, 'activation': 'relu', 'normalization': 'batch',
                       'conv_mode': 'same', 'dim': 2, 'up_mode': 'transposed'}
        for k, _ in unet_params.items():
            if k in model_params:
                unet_params[k] = model_params[k]

        model = UNet(**unet_params)
    else:
        print('No model',model_params['model_name'])
        assert True==False

    if model_params['freeze']:
        for param in model.parameters():
            param.requires_grad = False

    return model
