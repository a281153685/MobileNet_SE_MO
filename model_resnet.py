# -*- coding: utf-8 -*-
# @Author  : Mumu
# @Time    : 2021/11/16 18:18
'''
@Function:
 
'''
import torch.nn as nn
from torch.nn import init
from Model_util_ResNet.CBAM import *
import torch.nn.functional as F

def conv3(input_channel, output_channel, stride=1):
    return nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_channel, out_channel, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3(input_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(input_channel, out_channel, stride)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.stride = stride
        self.downsample = downsample

        if use_cbam:
            self.cbam = CBAM(out_channel, 16)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, input_channel, output_channel, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channel*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if use_cbam:
            self.cbam = CBAM(output_channel*4, 16)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, network_type, num_class, att_type=None):
        super(ResNet, self).__init__()
        self.input_channel = 64
        self.network_type = network_type
        if network_type == 'ImageNet':
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AvgPool2d(7)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        if att_type == 'CBAM':
            self.cbam1 = CBAM(64 * block.expansion, 16)
            self.cbam2 = CBAM(128 * block.expansion, 16)
            self.cbam3 = CBAM(256 * block.expansion, 16)
        else:
            self.cbam1 = None
            self.cbam2 = None
            self.cbam3 = None

        self.layer1 = self._make_layer(block, 64, layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type)

        self.fc = nn.Linear(512*block.expansion, num_class)
        init.kaiming_normal(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    if 'SpatialGate' in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, output_channel, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.input_channel != output_channel*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.input_channel, output_channel*block.expansion,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channel*block.expansion),
            )
        layers = []
        layers.append(block(self.input_channel, output_channel, stride, downsample, use_cbam=att_type=='CBAM'))
        self.input_channel = output_channel * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.input_channel, output_channel, use_cbam=att_type=='CBAM'))

        return nn.Sequential(*layers)

    def forward(self, x ):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == 'ImageNet':
            x = self.maxpool(x)

        x = self.layer1(x)
        if not self.cbam1 is None:
            x = self.cbam1(x)

        x = self.layer2(x)
        if not self.cbam2 is None:
            x = self.cbam2(x)

        x = self.layer3(x)
        if not self.cbam3 is None:
            x = self.cbam3(x)

        x = self.layer4(x)

        if self.network_type == 'ImageNet':
            x = self.avgpool(x)
        else:
            x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResidualNet(network_type, depth, num_classes, att_type):

    assert network_type in ['ImageNet', 'CIFAR10', "CIFAR100"], 'network tyepe should be ImageNet'
    assert depth in [18, 34, 50, 101], 'network depth should be 18,34,50 or 101'

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type)

    return model

