from functools import partial
from inspect import signature

import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from models.twod_models.common import TemporalPooling
from models.twod_models.temporal_modeling import temporal_modeling_module

__all__ = ['resnet']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, num_frames, stride=1, downsample=None, temporal_module=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.tam = temporal_module(duration=num_frames, channels=inplanes) \
            if temporal_module is not None else None

    def forward(self, x):
        identity = x
        if self.tam is not None:
            x = self.tam(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, num_frames, stride=1, downsample=None, temporal_module=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.tam = temporal_module(duration=num_frames, channels=inplanes) \
            if temporal_module is not None else None

    def forward(self, x):
        identity = x
        if self.tam is not None:
           x = self.tam(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
            
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_frames, num_classes=1000, dropout=0.5, zero_init_residual=False,
                 without_t_stride=False, temporal_module=None, pooling_method='max', modality='rgb'):
        super(ResNet, self).__init__()

        self.pooling_method = pooling_method.lower()
        block = BasicBlock if depth < 50 else Bottleneck
        layers = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]}[depth]
        self.depth = depth
        self.temporal_module = temporal_module
        self.num_frames = num_frames
        self.orig_num_frames = num_frames
        self.num_classes = num_classes

        if self.num_classes not in [5,8]:
            raise ValueError("Invalid number of classes: {}".format(self.num_classes))

        self.without_t_stride = without_t_stride
        self.modality = modality
        self.inplanes = 64
        if self.modality == 'rgb':
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif self.modality == 'gray':
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif self.modality == 'flow':
            self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            raise ValueError("Unknown modality {}".format(self.modality))            
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        if not self.without_t_stride:
            self.pool1 = TemporalPooling(self.num_frames, 3, 2, self.pooling_method)
            self.num_frames = self.num_frames // 2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        if not self.without_t_stride:
            self.pool2 = TemporalPooling(self.num_frames, 3, 2, self.pooling_method)
            self.num_frames = self.num_frames // 2
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        if not self.without_t_stride:
            self.pool3 = TemporalPooling(self.num_frames, 3, 2, self.pooling_method)
            self.num_frames = self.num_frames // 2
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(512 * block.expansion, 4)
        self.fc2 = nn.Linear(512 * block.expansion, 1)
        self.fc3 = nn.Linear(512 * block.expansion, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.num_frames, stride, downsample,
                            temporal_module=self.temporal_module))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.num_frames,
                                temporal_module=self.temporal_module))

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size, c_t, h, w = x.shape
        x = x.view(batch_size * self.orig_num_frames, c_t // self.orig_num_frames, h, w)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        fp1 = self.maxpool(x)

        fp2 = self.layer1(fp1)
        fp2_d = self.pool1(fp2) if not self.without_t_stride else fp2
        fp3 = self.layer2(fp2_d)
        fp3_d = self.pool2(fp3) if not self.without_t_stride else fp3
        fp4 = self.layer3(fp3_d)
        fp4_d = self.pool3(fp4) if not self.without_t_stride else fp4
        fp5 = self.layer4(fp4_d)

        
        x1 = self.avgpool(fp5)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.dropout(x1)
        x1 = self.fc1(x1)

        n_t, c = x1.shape
        out1 = x1.view(batch_size, -1, c)

        # average the prediction from all frames
        out1 = torch.mean(out1, dim=1)

        fp6 = self.layer4(fp4_d)
        x2 = self.avgpool(fp6)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.dropout(x2)
        x2 = self.fc2(x2)

        n_t, c = x2.shape
        out2 = x2.view(batch_size, -1, c)

        # average the prediction from all frames
        out2 = torch.mean(out2, dim=1)


        x3 = self.avgpool(fp5)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.dropout(x3)
        x3 = self.fc3(x3)

        n_t, c = x3.shape
        out3 = x3.view(batch_size, -1, c)

        # average the prediction from all frames
        out3 = torch.mean(out3, dim=1)

        return out1, out2, out3

    def mean(self, modality='rgb'):
        return [0.0, 0.0, 0.0] if modality == 'rgb' else [0.0]

    def std(self, modality='rgb'):
        return [1.0, 1.0, 1.0] if modality == 'rgb' else [1.0]

    @property
    def network_name(self):
        name = ''
        if self.temporal_module is not None:
            param = signature(self.temporal_module).parameters
            temporal_module = str(param['name']).split("=")[-1][1:-1]
            blending_frames = str(param['blending_frames']).split("=")[-1]
            blending_method = str(param['blending_method']).split("=")[-1][1:-1]
            dw_conv = True if str(param['dw_conv']).split("=")[-1] == 'True' else False
            name += "{}-b{}-{}{}-".format(temporal_module, blending_frames,
                                         blending_method,
                                         "" if dw_conv else "-allc")
        name += 'resnet-{}'.format(self.depth)
        if not self.without_t_stride:
            name += "-ts-{}".format(self.pooling_method)

        return name


def resnet_multihead(depth, num_classes, without_t_stride, groups, temporal_module_name,
           dw_conv, blending_frames, blending_method, dropout, pooling_method,
           imagenet_pretrained=True, modality='rgb', **kwargs):

    temporal_module = partial(temporal_modeling_module, name=temporal_module_name,
                              dw_conv=dw_conv,
                              blending_frames=blending_frames,
                              blending_method=blending_method) if temporal_module_name is not None \
        else None

    model = ResNet(depth, num_frames=groups, num_classes=num_classes,
                   without_t_stride=without_t_stride,
                   temporal_module=temporal_module, dropout=dropout,
                   pooling_method=pooling_method, modality=modality)

    if imagenet_pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet{}'.format(depth)], map_location='cpu')
        state_dict.pop('fc1.weight', None)
        state_dict.pop('fc1.bias', None)
        state_dict.pop('fc2.weight', None)
        state_dict.pop('fc2.bias', None)
        state_dict.pop('fc3.weight', None)
        state_dict.pop('fc3.bias', None)
        if modality == 'gray':
            state_dict['conv1.weight'] = torch.mean(state_dict['conv1.weight'], dim=1, keepdim=True)
        model.load_state_dict(state_dict, strict=False)
        print("============> loaded pre-trained model ")

    return model
