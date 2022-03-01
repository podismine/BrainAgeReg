import torch.nn as nn
import torch.nn.functional as F

class SFCN(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=40, dropout=True):
        super(SFCN, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
        self.classifier = nn.Sequential()

        avg_shape = [2, 3, 2]
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.4))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))
        self.f1 = nn.Conv3d(out_channel, out_channel //2, padding=0, kernel_size=1)
        self.f2 = nn.Conv3d(out_channel // 2, out_channel //4, padding=0, kernel_size=1)
        self.f3 = nn.Conv3d(out_channel // 4, 1, padding=0, kernel_size=1)
    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(inplace = True),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(inplace = True)
            )
        return layer

    def forward(self, x, is_regression = False):

        x_f = self.feature_extractor(x)
        x1 = self.classifier(x_f)
        x2 = self.f1(x1)
        x3 = self.f2(x2)
        if is_regression is True:
            out_logits = self.f3(x3).view(x3.size(0))
            out = x3
        else:
            out_logits = F.log_softmax(x3.view(x3.size(0), -1), dim=1)
            out = x3.view(x3.size(0), -1)

        return out_logits, out, x1


