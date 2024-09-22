"""
v1 uses the model architecture from paper FLANC (original Resent: post norm):
https://github.com/HarukiYqM/All-In-One-Neural-Composition
v2 uses the model architecture from paper FedRolex (new Resnet: pre norm):
https://github.com/AIoT-MLSys-Lab/FedRolex
To keep with the other baselines, the only change I make is setting:
    nn.BatchNorm2d(channels)--->nn.BatchNorm2d(channels, momentum=0., track_running_stats=None)
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from model.utils import init_param, Scaler

def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def branchBottleNeck(channel_in, channel_out, kernel_size):
    return nn.Sequential(
        nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride=2, padding=1),
        nn.BatchNorm2d(channel_out, momentum=0.0, track_running_stats=None),
        nn.ReLU(),
    )

def Adapter(in_planes, out_planes, stride=1):
    """
    3x3 conolution with padding, bn layer and Relu
    """
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_planes, momentum=0.0, track_running_stats=None),
        nn.ReLU(),
    )

def decom_conv(in_channels, out_channels, kernel_size=3, stride=1, bias=True):
    m = nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias,
    )
    if kernel_size == 3:
        torch.nn.init.orthogonal_(m.weight)
    return m

class FactorizedConv(nn.Module):

    def __init__(self, in_channels, out_channels, n_basis,
                 stride=1, bias=False, de_conv=decom_conv):
        super(FactorizedConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        modules = [nn.Conv2d(in_channels, n_basis, kernel_size=(1, 3),
            padding=(0, 1), stride=(1, stride), dilation=(1, 1), bias=bias,
        )]
        modules.append(nn.Conv2d(n_basis, out_channels, kernel_size=(3, 1),
            padding=(1, 0), stride=(stride, 1), dilation=(1, 1), bias=bias,
        ))
        self.conv = nn.Sequential(*modules)

    def recover(self):
        conv1 = self.conv[0] # (rank, inplanes, 1, 3)
        conv1.weight.data = conv1.weight.data.permute(1, 3, 2, 0)
        a, b, c, d = conv1.weight.shape
        dim1, dim2 = a * b, c * d
        VT = conv1.weight.data.reshape(dim1, dim2)
        conv2 = self.conv[1] # (outplanes, rank, 3, 1)
        conv2.weight.data = conv2.weight.data.permute(0, 2, 1, 3)
        a, b, c, d = conv2.weight.shape
        dim1, dim2 = a * b, c * d
        U = conv2.weight.data.reshape(dim1, dim2)
        W = torch.matmul(U, VT.T).reshape(self.out_channels, 3, self.in_channels, 3,).permute(0, 2, 1, 3)
        return W

    def frobenius_loss(self):
        conv1 = self.conv[0]
        conv2 = self.conv[1]
        temp_VT = conv1.weight.permute(1, 3, 2, 0)
        a, b, c, d = temp_VT.data.shape
        dim1, dim2 = a * b, c * d
        VT = torch.reshape(temp_VT, (dim1, dim2))
        temp_UT = conv2.weight.permute(0, 2, 1, 3)
        a, b, c, d = temp_UT.data.shape
        dim1, dim2 = a * b, c * d
        U = torch.reshape(temp_UT, (dim1, dim2))
        loss = torch.norm(torch.matmul(U, torch.transpose(VT, 0, 1)), p='fro')**2
        return loss

    def L2_loss(self):
        conv1 = self.conv[0]
        conv2 = self.conv[1]
        loss = torch.norm(conv1.weight, p='fro')**2 + torch.norm(conv2.weight, p='fro')**2
        return loss

    def kronecker_loss(self):
        conv1 = self.conv[0]
        conv2 = self.conv[1]
        loss = (torch.norm(conv1.weight, p='fro')**2) * (torch.norm(conv2.weight, p='fro')**2)
        return loss



    def forward(self, x):
        return self.conv(x)

class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0, rate=1, track=None, cfg=None, ):
        super(BasicBlock, self).__init__()

        self.inplanes = inplanes
        self.outplanes = planes

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.0, track_running_stats=track)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.0, track_running_stats=track)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(p=dropout_rate)
        if cfg['scale']:
            self.scaler = Scaler(rate)
        else:
            self.scaler = nn.Identity()

    def decom(self, ratio_LR):
        self.rank = round(ratio_LR * self.outplanes)
        a, b, c, d = self.conv1.weight.shape  # (outplanes, inplanes, k, k)
        dim1, dim2 = a * c, b * d
        W = self.conv1.weight.data.reshape(dim1, dim2)
        U, S, V = torch.svd(W)
        sqrtS = torch.diag(torch.sqrt(S[:self.rank]))
        new_U, new_V = torch.matmul(U[:, :self.rank], sqrtS), torch.matmul(V[:, :self.rank], sqrtS).T
        self.conv1 = FactorizedConv(self.inplanes, self.outplanes, self.rank, stride=self.stride, bias=False)
        self.conv1.conv[0].weight.data = new_V.reshape(self.inplanes, c, 1, self.rank).permute(3, 0, 2, 1)
        self.conv1.conv[1].weight.data = new_U.reshape(self.outplanes, c, self.rank, 1).permute(0, 2, 1, 3)

        a, b, c, d = self.conv2.weight.shape
        dim1, dim2 = a * c, b * d
        W = self.conv2.weight.data.reshape(dim1, dim2)
        U, S, V = torch.svd(W)
        sqrtS = torch.diag(torch.sqrt(S[:self.rank]))
        new_U, new_V = torch.matmul(U[:, :self.rank], sqrtS), torch.matmul(V[:, :self.rank], sqrtS).T
        self.conv2 = FactorizedConv(self.outplanes, self.outplanes, self.rank, stride=1, bias=False)
        self.conv2.conv[0].weight.data = new_V.reshape(self.outplanes, c, 1, self.rank).permute(3, 0, 2, 1)
        self.conv2.conv[1].weight.data = new_U.reshape(self.outplanes, c, self.rank, 1).permute(0, 2, 1, 3)
        # print("Done")

    def recover(self):
        W1 = self.conv1.recover()
        W2 = self.conv2.recover()
        self.conv1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.conv1.weight.data = W1
        self.conv2 = nn.Conv2d(self.outplanes, self.outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2.weight.data = W2

    def forward(self, x):
        residual = x
        out = self.scaler(self.conv1(x))
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.scaler(self.conv2(out))
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, rate=1, track=None, cfg=None,):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.0, track_running_stats=track)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.0, track_running_stats=track)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4, momentum=0.0, track_running_stats=track)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if cfg['scale']:
            self.scaler = Scaler(rate)
        else:
            self.scaler = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.scaler(self.conv1(x))
        out = self.bn1(out)
        out = self.relu(out)
        out = self.scaler(self.conv2(out))
        out = self.bn2(out)
        out = self.relu(out)
        out = self.scaler(self.conv3(out))
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, data_shape, hidden_size, block, num_blocks, num_classes,
                 depth_rate, rate, track, cfg, dropout_rate=0,):
        super(ResNet, self).__init__()

        self.cfg = cfg
        self.dataset_name = cfg['dataset_name']
        self.depth_rate = depth_rate
        self.in_planes = hidden_size[0]
        self.dropout_rate = dropout_rate
        self.feature_num = hidden_size[-1]
        self.class_num = num_classes

        # if self.dataset_name == 'tinyImagenet':
        #     self.head = nn.Sequential(
        #         nn.Conv2d(data_shape[0], self.in_planes, kernel_size=7, stride=2, padding=3, bias=False),
        #         nn.BatchNorm2d(self.in_planes, momentum=0.0, track_running_stats=track),
        #     )
        #     self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # else:
        #     self.head = nn.Sequential(
        #         nn.Conv2d(data_shape[0], self.in_planes, kernel_size=3, stride=1, padding=1, bias=False),
        #         nn.BatchNorm2d(self.in_planes, momentum=0.0, track_running_stats=track),
        #     )

        self.head = nn.Sequential(
            nn.Conv2d(data_shape[0], self.in_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_planes, momentum=0.0, track_running_stats=track),
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1, rate=rate, track=track)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2, rate=rate, track=track)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2, rate=rate, track=track)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2, rate=rate, track=track)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.tail = nn.Linear(self.feature_num, self.class_num)
        self.body = nn.Sequential(*[self.layer1, self.layer2, self.layer3, self.layer4])

        if cfg['scale']:
            self.scaler = Scaler(rate)
        else:
            self.scaler = nn.Identity()
        # initial parameter
        """https://github.com/tding1/Neural-Collapse/blob/main/models/resnet.py"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def recover_large_layer(self, layer_index):
        for i, idx in enumerate(layer_index):
            meta_block = self.body.pop(idx)
            for j in range(2):
                meta_block[j].recover()
            # meta_block_params = copy.deepcopy(meta_block.state_dict())
            self.body.insert(idx, meta_block)

    def decom_large_layer(self, layer_index, ratio_LR=0.2):
        for i, idx in enumerate(layer_index):
            large_block = self.body.pop(idx)
            for j in range(2):
                large_block[j].decom(ratio_LR=ratio_LR)
            # large_block_params = copy.deepcopy(large_block.state_dict())
            self.body.insert(idx, large_block)

    def _make_layer(self, block, planes, blocks, stride=1, rate=1, track=None):
        cfg = self.cfg
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.0, track_running_stats=track)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample=downsample, dropout_rate=self.dropout_rate, rate=rate, track=track, cfg=cfg))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, dropout_rate=self.dropout_rate, rate=rate, track=track, cfg=cfg))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.head(x)
        x = self.relu(x)
        # if self.dataset_name == 'tinyImagenet':
        #     x = self.maxpool(x)
        for layer in self.body:
            x = layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.tail(x)

def resnet18(model_rate=1, depth_rate=4, track=False, cfg=None):
    """
    :param model_rate:
    :param depth_rate:
    :param track:
    :param cfg:
    :return:
    """
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    hidden_size = [int(np.ceil(model_rate * x)) for x in cfg['resnet']['hidden_size']]
    scaler_rate = model_rate / cfg['global_model_rate']
    model = ResNet(data_shape, hidden_size, BasicBlock, [2, 2, 2, 2], classes_size, depth_rate, scaler_rate, track, cfg)
    # model.apply(init_param)
    return model


def resnet34(model_rate=1, depth_rate=None, track=False, cfg=None):
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    hidden_size = [int(np.ceil(model_rate * x)) for x in cfg['resnet']['hidden_size']]
    scaler_rate = model_rate / cfg['global_model_rate']
    model = ResNet(data_shape, hidden_size, BasicBlock, [3, 4, 6, 3], classes_size, depth_rate, scaler_rate, track, cfg)
    model.apply(init_param)
    return model





