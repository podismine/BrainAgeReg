import torch
import torch.nn as nn

class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Reduction_A(nn.Module):
    # 35 -> 17
    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch_0 = Conv3d(in_channels, n, 3, stride=2, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, k, 1, stride=1, padding=0, bias=False),
            Conv3d(k, l, 3, stride=1, padding=1, bias=False),
            Conv3d(l, m, 3, stride=2, padding=0, bias=False),
        )
        self.branch_2 = nn.MaxPool3d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1) # 17 x 17 x 1024

class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.features = nn.Sequential(
            Conv3d(in_channels, 32, 3, stride=2, padding=0, bias=False), # 149 x 149 x 32
            Conv3d(32, 32, 3, stride=1, padding=0, bias=False), # 147 x 147 x 32
            Conv3d(32, 64, 3, stride=1, padding=1, bias=False), # 147 x 147 x 64
            nn.MaxPool3d(3, stride=2, padding=0), # 73 x 73 x 64
            Conv3d(64, 80, 1, stride=1, padding=0, bias=False), # 73 x 73 x 80
            Conv3d(80, 192, 3, stride=1, padding=0, bias=False), # 71 x 71 x 192
            nn.MaxPool3d(3, stride=2, padding=0), # 35 x 35 x 192
        )
        self.branch_0 = Conv3d(192, 96, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv3d(192, 48, 1, stride=1, padding=0, bias=False),
            Conv3d(48, 64, 5, stride=1, padding=2, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv3d(192, 64, 1, stride=1, padding=0, bias=False),
            Conv3d(64, 96, 3, stride=1, padding=1, bias=False),
            Conv3d(96, 96, 3, stride=1, padding=1, bias=False),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool3d(3, stride=1, padding=1, count_include_pad=False),
            Conv3d(192, 64, 1, stride=1, padding=0, bias=False)
        )
    def forward(self, x):
        x = self.features(x)
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Inception_ResNet_A(nn.Module):
    def __init__(self, in_channels, scale=1.0):
        super(Inception_ResNet_A, self).__init__()
        self.scale = scale
        self.branch_0 = Conv3d(in_channels, 32, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Conv3d(32, 32, 3, stride=1, padding=1, bias=False)
        )
        self.branch_2 = nn.Sequential(
            Conv3d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Conv3d(32, 48, 3, stride=1, padding=1, bias=False),
            Conv3d(48, 64, 3, stride=1, padding=1, bias=False)
        )
        self.conv = nn.Conv3d(128, 320, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x_res = torch.cat((x0, x1, x2), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)


class Inception_ResNet_B(nn.Module):
    def __init__(self, in_channels, scale=1.0):
        super(Inception_ResNet_B, self).__init__()
        self.scale = scale
        self.branch_0 = Conv3d(in_channels, 192, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, 128, 1, stride=1, padding=0, bias=False),
            Conv3d(128, 160, (1, 7, 1), stride=1, padding=(0, 3, 0), bias=False),
            Conv3d(160, 192, (7, 1, 1), stride=1, padding=(3, 0, 0), bias=False),
            #Conv3d(192, 192, (1, 1, 7), stride=1, padding=(0, 0, 7), bias=False)
        )
        self.conv = nn.Conv3d(384, 1088, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)


class Reduciton_B(nn.Module):
    def __init__(self, in_channels):
        super(Reduciton_B, self).__init__()
        self.branch_0 = nn.Sequential(
            Conv3d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Conv3d(256, 384, 3, stride=2, padding=0, bias=False)
        )
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Conv3d(256, 288, 3, stride=2, padding=0, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv3d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Conv3d(256, 288, 3, stride=1, padding=1, bias=False),
            Conv3d(288, 320, 3, stride=2, padding=0, bias=False)
        )
        self.branch_3 = nn.MaxPool3d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Inception_ResNet_C(nn.Module):
    def __init__(self, in_channels, scale=1.0, activation=True):
        super(Inception_ResNet_C, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = Conv3d(in_channels, 192, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            Conv3d(192, 224, (1, 3,1), stride=1, padding=(0, 1, 0), bias=False),
            Conv3d(224, 256, (3, 1,1), stride=1, padding=(1, 0, 0), bias=False),
            #Conv3d(256, 256, (1, 1,3), stride=1, padding=(0, 0, 1), bias=False)
        )
        self.conv = nn.Conv3d(448, 2080, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        if self.activation:
            return self.relu(x + self.scale * x_res)
        return x + self.scale * x_res

import torch.nn.functional as F
class DBN(nn.Module):
    def __init__(self, in_channels=1, output_dim=88, k=256, l=256, m=384, n=384, mode = 0):
        super(DBN, self).__init__()
        self.mode = mode
        blocks = []
        blocks.append(Stem(in_channels))
        for i in range(10):
            blocks.append(Inception_ResNet_A(320, 0.17))
        blocks.append(Reduction_A(320, k, l, m, n))
        for i in range(20):
            blocks.append(Inception_ResNet_B(1088, 0.10))
        blocks.append(Reduciton_B(1088))
        for i in range(9):
            blocks.append(Inception_ResNet_C(2080, 0.20))
        blocks.append(Inception_ResNet_C(2080, activation=False))
        self.features = nn.Sequential(*blocks)
        self.conv = Conv3d(2080, 1536, 1, stride=1, padding=0, bias=False)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1536, output_dim)
        self.f1 = nn.Linear(output_dim, output_dim // 2)
        self.f2 = nn.Linear(output_dim // 2, output_dim // 4)
        self.f3 = nn.Linear(output_dim // 4, 1)

    def forward(self, x, is_regression = False):
        x = self.features(x)
        x = self.conv(x)
        x = self.global_average_pooling(x)
        x = x.view(x.size(0), -1)
        rep = self.linear(x)
        x = self.f1(rep)
        x = self.f2(x)

        if is_regression is True:
            out_logits = self.f3(x).view(x.size(0))
            out = x
        else:
            out_logits = F.log_softmax(x.view(x.size(0), -1), dim=1)
            out = x.view(x.size(0), -1)
        return out_logits, out, rep