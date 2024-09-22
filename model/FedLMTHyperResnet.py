

import numpy as np
import torch
import torch.nn as nn
from model.resnetv1 import BasicBlock, Bottleneck, decom_conv, FactorizedConv
from model.utils import init_param, Scaler


class MetaBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, n_basis=1, downsample=None,
                 dropout_rate=0, rate=1, track=None, cfg=None, ):
        super(MetaBasicBlock, self).__init__()

        self.cfg = cfg
        self.inplanes = inplanes
        self.outplanes = planes
        self.rank = n_basis
        self.stride = stride

        # self.conv1 = DecomBlock(inplanes, planes, n_basis, stride=stride, bias=False)
        self.conv1 = FactorizedConv(inplanes, planes, n_basis, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.0, track_running_stats=track)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = DecomBlock(planes, planes, n_basis, stride=1, bias=False)
        self.conv2 = FactorizedConv(planes, planes, n_basis, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.0, track_running_stats=track)
        self.downsample = downsample

        self.dropout = nn.Dropout(p=dropout_rate)
        if cfg['scale']:
            self.scaler = Scaler(rate)  # rate=1���S�RI�p
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

    def cal_smallest_svdvals(self):
        """
        calculate the smallest singular value of a residual block
        """

        temp_VT = self.conv1.conv[0].weight.data.clone().permute(1, 3, 2, 0)
        a, b, c, d = temp_VT.shape
        dim1, dim2 = a * b, c * d
        V1 = torch.reshape(temp_VT, (dim1, dim2))

        temp_UT = self.conv1.conv[1].weight.data.clone().permute(0, 2, 1, 3)
        a, b, c, d = temp_UT.shape
        dim1, dim2 = a * b, c * d
        U1 = torch.reshape(temp_UT, (dim1, dim2))

        S1 = torch.linalg.svdvals(V1)
        S2 = torch.linalg.svdvals(U1)

        temp_VT = self.conv2.conv[0].weight.data.clone().permute(1, 3, 2, 0)
        a, b, c, d = temp_VT.shape
        dim1, dim2 = a * b, c * d
        V2 = torch.reshape(temp_VT, (dim1, dim2))

        temp_UT = self.conv2.conv[1].weight.data.clone().permute(0, 2, 1, 3)
        a, b, c, d = temp_UT.shape
        dim1, dim2 = a * b, c * d
        U2 = torch.reshape(temp_UT, (dim1, dim2))

        S3 = torch.linalg.svdvals(V2)
        S4 = torch.linalg.svdvals(U2)

        S = torch.cat([S1, S2, S3, S4], dim=0)
        return torch.min(S).item()

    def recover(self):
        W1 = self.conv1.recover()
        W2 = self.conv2.recover()
        self.conv1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.conv1.weight.data = W1
        self.conv2 = nn.Conv2d(self.outplanes, self.outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2.weight.data = W2

    def frobenius_loss(self):
        loss1 = self.conv1.frobenius_loss()
        loss2 = self.conv2.frobenius_loss()
        return (loss1 + loss2)

    def kronecker_loss(self):
        loss1 = self.conv1.kronecker_loss()
        loss2 = self.conv2.kronecker_loss()
        return (loss1 + loss2)

    def L2_loss(self):
        loss1 = self.conv1.L2_loss()
        loss2 = self.conv2.L2_loss()
        return (loss1 + loss2)

    def forward(self, x):
        residual = x

        out = self.scaler(self.conv1(x))
        out = self.bn1(out)
        out = self.relu(out)

        out = self.scaler(self.conv2(out))
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PFLResnet(nn.Module):
    def __init__(self, data_shape, hidden_size, block, num_blocks, personalized_rule, num_classes=10,
                 rate=1, track=None, cfg=None, dropout_rate=0, ):
        super(PFLResnet, self).__init__()
        """
        decom_rule is a 2-tuple like (block_index, layer_index).
        For resnet18, block_index is selected from [0,1,2,3] and layer_index is selected from [0,1].
        Example: If we only want to decompose layers starting form the 8-th layer for resnet18, 
                 then we set decom_rule = (1, 1);
                 If we want to decompose all layer(except head and tail layer), we can set 
                 decom_rule = (-1, 0);
                 If we don't want to decompose any layer, we can set 
                 decom_rule = (4, 0).
        """
        self.cfg = cfg
        self.dataset_name = cfg['dataset_name']
        self.personalized_rule = personalized_rule

        self.inplanes = hidden_size[0]
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.feature_num = hidden_size[-1]
        self.class_num = num_classes

        self.head = nn.Sequential(
            nn.Conv2d(data_shape[0], self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=None, track_running_stats=None),
        )
        self.relu = nn.ReLU(inplace=True)

        # initialization the model
        strides = [1, 2, 2, 2]
        body_layers, common_layers, personalized_layers = [], [], []
        common_layers.append(self.head)
        for block_idx in range(4):
            layer = self._make_larger_layer(block=block, planes=hidden_size[block_idx], blocks=num_blocks[block_idx],
                                            stride=strides[block_idx], rate=rate, track=track)
            body_layers.append(layer)
            if block_idx < self.personalized_rule[0]:
                common_layers.append(layer)
            elif block_idx == self.personalized_rule[0]:
                for layer_idx in range(self.personalized_rule[1]):
                    common_layers.append(layer[layer_idx])
                for layer_idx in range(self.personalized_rule[1], self.num_blocks[block_idx]):
                    personalized_layers.append(layer[layer_idx])
            else:
                personalized_layers.append(layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.tail = nn.Linear(self.feature_num, num_classes)
        personalized_layers.append(self.tail)

        self.common = nn.Sequential(*common_layers)
        self.personalized = nn.Sequential(*personalized_layers)
        self.body = nn.Sequential(*body_layers)

        # initialization for the hybrid model parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_larger_layer(self, block, planes, blocks, stride=1, rate=1, track=None):
        cfg = self.cfg
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.0, track_running_stats=track)
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample=downsample, dropout_rate=self.dropout_rate, rate=rate,
                  track=track, cfg=cfg))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dropout_rate=self.dropout_rate, rate=rate, track=track, cfg=cfg))

        return nn.Sequential(*layers)

    def forward(self, x, ):
        x = self.head(x)
        x = self.relu(x)
        # if self.dataset_name == 'tinyImagenet':
        #     x = self.maxpool(x)
        for layer in self.body:
            x = layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.tail(x)

        return x


class HyperResNet(nn.Module):

    def __init__(self, data_shape, hidden_size, block, num_blocks, ratio_LR, decom_rule, num_classes=10,
                 rate=1, track=None, cfg=None, dropout_rate=0, ):
        super(HyperResNet, self).__init__()
        """
        decom_rule is a 2-tuple like (block_index, layer_index).
        For resnet18, block_index is selected from [0,1,2,3] and layer_index is selected from [0,1].
        Example: If we only want to decompose layers starting form the 8-th layer for resnet18, 
                 then we set decom_rule = (1, 1);
                 If we want to decompose all layer(expept head and tail layer), we can set 
                 decom_rule = (-1, 0);
                 If we don't want to decompose any layer, we can set 
                 decom_rule = (4, 0).
        """
        self.cfg = cfg
        self.dataset_name = cfg['dataset_name']
        self.decom_rule = decom_rule

        self.inplanes = hidden_size[0]
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.feature_num = hidden_size[-1]
        self.class_num = num_classes
        self.ratio_LR = ratio_LR

        self.head = nn.Sequential(
            nn.Conv2d(data_shape[0], self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=None, track_running_stats=None),
        )
        self.relu = nn.ReLU(inplace=True)

        # initialization the hybrid model
        strides = [1, 2, 2, 2]
        all_layers, common_layers, personalized_layers = [], [], []
        common_layers.append(self.head)
        for block_idx in range(4):
            if block_idx < self.decom_rule[0]:
                layer = self._make_larger_layer(block=block, planes=hidden_size[block_idx],
                                                blocks=num_blocks[block_idx],
                                                stride=strides[block_idx], rate=rate, track=track)
                all_layers.append(layer)
                common_layers.append(layer)
            elif block_idx == self.decom_rule[0]:
                config = round(hidden_size[block_idx] * self.ratio_LR)  # rank
                layer = self._make_hybrid_layer(large_block=block, meta_block=MetaBasicBlock,
                                                planes=hidden_size[block_idx],
                                                blocks=num_blocks[block_idx], stride=strides[block_idx],
                                                start_decom_idx=self.decom_rule[1], config=config,
                                                rate=rate, track=track, )
                all_layers.append(layer)
                for layer_idx in range(self.decom_rule[1]):
                    common_layers.append(layer[layer_idx])
                for layer_idx in range(self.decom_rule[1], self.num_blocks[block_idx]):
                    personalized_layers.append(layer[layer_idx])

            elif block_idx > self.decom_rule[0]:
                config = round(hidden_size[block_idx] * self.ratio_LR)  # rank
                layer = self._make_meta_layer(block=MetaBasicBlock, planes=hidden_size[block_idx],
                                              blocks=num_blocks[block_idx],
                                              config=config, stride=strides[block_idx], rate=rate, track=track)
                all_layers.append(layer)
                personalized_layers.append(layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.tail = nn.Linear(self.feature_num, num_classes)
        personalized_layers.append(self.tail)

        self.body = nn.Sequential(*all_layers)
        self.common = nn.Sequential(*common_layers)
        self.personalized = nn.Sequential(*personalized_layers)

        # initialization for the hybrid model parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_meta_layer(self, block, planes, blocks, config=None, stride=1, rate=1, track=None):
        cfg = self.cfg
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.0, track_running_stats=track)
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride=stride, n_basis=config, downsample=downsample,
                            dropout_rate=self.dropout_rate, rate=rate, track=track, cfg=cfg))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, stride=1, n_basis=config, dropout_rate=self.dropout_rate, rate=rate,
                      track=track, cfg=cfg))
        return nn.Sequential(*layers)

    def _make_hybrid_layer(self, large_block, meta_block, planes, blocks, stride=1, rate=1, start_decom_idx=0,
                           config=1, track=None):
        """
        :param start_decom_idx: range from [0, blocks-1]
        """
        cfg = self.cfg
        downsample = None
        block = meta_block if start_decom_idx == 0 else large_block

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.0, track_running_stats=track)
            )
        layers = []
        if start_decom_idx == 0:
            block = meta_block
            layers.append(
                block(self.inplanes, planes, stride, downsample=downsample, dropout_rate=self.dropout_rate, rate=rate,
                      n_basis=config, track=track, cfg=cfg))
        else:
            block = large_block
            layers.append(
                block(self.inplanes, planes, stride, downsample=downsample, dropout_rate=self.dropout_rate, rate=rate,
                      track=track, cfg=cfg))
        self.inplanes = planes * block.expansion

        for idx in range(1, blocks):
            block = large_block if idx < start_decom_idx else meta_block
            if idx < start_decom_idx:
                block = large_block
                layers.append(
                    block(self.inplanes, planes, dropout_rate=self.dropout_rate, rate=rate, track=track, cfg=cfg))
            else:
                block = meta_block
                layers.append(
                    block(self.inplanes, planes, dropout_rate=self.dropout_rate, rate=rate, n_basis=config, track=track,
                          cfg=cfg))

        return nn.Sequential(*layers)


    def _make_larger_layer(self, block, planes, blocks, stride=1, rate=1, track=None):
        cfg = self.cfg
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.0, track_running_stats=track)
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample=downsample, dropout_rate=self.dropout_rate, rate=rate,
                  track=track, cfg=cfg))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dropout_rate=self.dropout_rate, rate=rate, track=track, cfg=cfg))

        return nn.Sequential(*layers)


    def recover_large_layer(self, ):
        length = len(self.personalized)
        for idx, block in enumerate(self.personalized):
            # the last part of self.personalized is linear layer which is not decomposed
            if idx < length - 1:
                if isinstance(block, MetaBasicBlock):
                    block.recover()
                else:
                    for j in range(len(block)):
                        block[j].recover()


    def decom_large_layer(self, ratio_LR=0.2):
        length = len(self.personalized)
        for idx, block in enumerate(self.personalized):
            # the last part of self.personalized is linear layer which is not decomposed
            if idx < length - 1:
                if isinstance(block, MetaBasicBlock):
                    block.decom(ratio_LR=ratio_LR)
                else:
                    for j in range(len(block)):
                        block[j].decom(ratio_LR=ratio_LR)


    def frobenius_decay(self):
        loss = torch.tensor(0.).to(self.cfg['device'])
        length = len(self.personalized)
        for idx, block in enumerate(self.personalized):
            # the last part of self.personalized is linear layer which is not decomposed
            if idx < length - 1:
                if isinstance(block, MetaBasicBlock):
                    loss += block.frobenius_loss()
                else:
                    for j in range(len(block)):
                        loss += block[j].frobenius_loss()
        return loss


    def kronecker_decay(self):
        loss = torch.tensor(0.).to(self.cfg['device'])
        length = len(self.personalized)
        for idx, block in enumerate(self.personalized):
            # the last part of self.personalized is linear layer which is not decomposed
            if idx < length - 1:
                if isinstance(block, MetaBasicBlock):
                    loss += block.kronecker_loss()
                else:
                    for j in range(len(block)):
                        loss += block[j].kronecker_loss()
        return loss


    def L2_decay(self):
        loss = torch.tensor(0.).to(self.cfg['device'])
        length = len(self.personalized)
        for idx, block in enumerate(self.personalized):
            # the last part of self.personalized is linear layer which is not decomposed
            if idx < length - 1:
                if isinstance(block, MetaBasicBlock):
                    loss += block.L2_loss()
                else:
                    for j in range(len(block)):
                        loss += block[j].L2_loss()
        return loss


    def cal_smallest_svdvals(self):
        """
        calculate the smallest singular value of each residual block.
        For example, if the model is resnet18, then there are 8 residual blocks.
        """
        smallest_svdvals = []
        length = len(self.personalized)
        for idx, block in enumerate(self.personalized):
            # the last part of self.personalized is linear layer which is not decomposed
            if idx < length - 1:
                if isinstance(block, MetaBasicBlock):
                    smallest_svdvals.append(block.cal_smallest_svdvals())
                else:
                    for j in range(len(block)):
                        smallest_svdvals.append(block[j].cal_smallest_svdvals())
        return smallest_svdvals


    def forward(self, x, ):
        x = self.head(x)
        x = self.relu(x)
        # if self.dataset_name == 'tinyImagenet':
        #     x = self.maxpool(x)
        for idx, layer in enumerate(self.body):
            x = layer(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.tail(x)

        return x


def pfl_resnet18(model_rate=1, personalized_rule=[1, 1], track=False, cfg=None):
    """
    :param model_rate:
    :param track:
    :param cfg:
    :return:
    """
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    hidden_size = [int(np.ceil(model_rate * x)) for x in cfg['resnet']['hidden_size']]
    scaler_rate = model_rate / cfg['global_model_rate']
    model = PFLResnet(data_shape, hidden_size, BasicBlock, [2, 2, 2, 2], personalized_rule=personalized_rule,
                      num_classes=classes_size, rate=scaler_rate, track=track, cfg=cfg)
    # model.apply(init_param)
    return model


def hybrid_resnet18(ratio_LR=1, decom_rule=[1, 1], track=False, cfg=None):

    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    hidden_size = cfg['resnet']['hidden_size']
    model = HyperResNet(data_shape, hidden_size, BasicBlock, [2, 2, 2, 2], ratio_LR=ratio_LR,
                        decom_rule=decom_rule, num_classes=classes_size, track=track, cfg=cfg)
    return model


def hybrid_resnet34(model_rate=1, ratio_LR=1, decom_rule=[1, 1], track=False, cfg=None):
    """
    :param model_rate:
    :param track:
    :param cfg:
    :return:
    """
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    hidden_size = [int(np.ceil(model_rate * x)) for x in cfg['resnet']['hidden_size']]
    scaler_rate = model_rate / cfg['global_model_rate']
    model = HyperResNet(data_shape, hidden_size, BasicBlock, [3, 4, 6, 3], ratio_LR=ratio_LR, decom_rule=decom_rule,
                        num_classes=classes_size, rate=scaler_rate, track=track, cfg=cfg)
    # model.apply(init_param)
    return model
