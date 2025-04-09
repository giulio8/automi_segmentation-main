import numpy as np
import torch
from torch import nn


class ArgmaxNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_labels):
        super(ArgmaxNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_labels = n_labels
        self.nets = None

    # INPUT STRUCTURE IS SAME OF NET STRUCTURE: DICT FROM ROI TO INPUT IMAGE
    def forward(self, x):
        c = self.nets["1"](x["1"])
        for organ in range(len(x.keys())):
            organ = str(organ + 1)
            t2 = self.nets[organ](x[organ])  # predict with the best performing net, for each organ
            c = t2 if c is None else torch.cat([c, t2], dim=1)  # concat the results

        # perform argmax
        output_masks = c.cpu().detach()
        matrix_shape = np.shape(output_masks[0])
        combination_matrix = np.zeros(shape=matrix_shape)

        output_masks[not np.argmax(output_masks)] = 0
        output_masks[output_masks >= 0.5] = 1
        output_masks[output_masks != 1] = 0
        return output_masks

    def initialize(self, nets):
        self.nets = nn.ModuleDict(nets)  # dict {1:net, 2:net, ..}

    def set_parameter_requires_grad(model):
        for param in model.parameters():
            param.requires_grad = False
