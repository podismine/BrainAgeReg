'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.functional as F
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, output_dim = 88, mode = 0):
        super(VGG, self).__init__()
        self.mode = mode
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(13824, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, output_dim),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels * m.kernel_size[2]
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

        self.f1 = nn.Linear(output_dim, output_dim // 2)
        self.f2 = nn.Linear(output_dim // 2, output_dim // 4)
        self.f3 = nn.Linear(output_dim // 4, 1)
    def forward(self, x, is_regression = False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        rep = self.classifier(x)
        x = self.f1(rep)
        x = self.f2(x)

        if is_regression is True:
            out_logits = self.f3(x).view(x.size(0))
            out = x
        else:
            out_logits = F.log_softmax(x.view(x.size(0), -1), dim=1)
            out = x.view(x.size(0), -1)

        return out_logits, out, rep
            


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg16_bn(output_dim, mode):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True), output_dim, mode)
