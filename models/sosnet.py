import torch
import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel * reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # tweak this to get batch and channels dims
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SOSnet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SOSnet, self).__init__()
        self.name = "Small-Organ-Segmentation-Net"
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.seLayer = SELayer(1)
        self.conv2d = nn.Conv2d(in_channels=n_channels, out_channels=n_classes, kernel_size=(1, 1))

    def forward(self, x):
        x1 = self.seLayer(x)
        x2 = self.seLayer(x1)
        return self.conv2d(x2)