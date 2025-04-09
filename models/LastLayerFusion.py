import torch
from torch import nn
from torch.nn import UpsamplingBilinear2d


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)


# classes = 7 (includes background)
# channel = 1
# labels = 6
# in features = 64*num_of_unet_or_seresunet + 256*num_of_deeplab
# nets = dict from roi to pretrained net on that roi
class LastLayerFusionNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_labels, in_features):
        super(LastLayerFusionNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_labels = n_labels
        self.nets = None
        self.outc = OutConv(in_features, n_classes)

    # INPUT STRUCTURE IS SAME OF NET STRUCTURE: DICT FROM ROI TO INPUT IMAGE
    def forward(self, x):
        c = None
        for organ in x.keys():
            # deeplab case
            if organ == "6":
                model = self.nets[organ]
                model.segmentation_head = UpsamplingBilinear2d(scale_factor=8.0)
                t2 = model(x[organ])
            else:
                t2 = self.nets[organ](x[organ])  # predict with the best performing net, for each organ
            c = t2 if c is None else torch.cat([c, t2], dim=1)  # concat the results

        x0 = self.outc(c)
        return x0

    def initialize(self, nets):
        self.nets = nn.ModuleDict(nets)  # dict {1:net, 2:net, ..}


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False
