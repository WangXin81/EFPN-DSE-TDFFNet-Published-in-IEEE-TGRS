
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import models 
from torch.autograd import Variable
import numpy as np

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

class BasicBlock(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(BasicBlock,self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(inplanes,64,kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,outplanes,kernel_size=1,stride=1,padding=0)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(inplanes,outplanes,kernel_size=1,stride=1,padding=0)
        self.bn4 = nn.BatchNorm2d(outplanes)

    def forward(self,x):
        identify = x
        identify = self.conv4(identify)
        identify = self.bn4(identify)
        identify = self.relu(identify)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identify
        out = self.relu(out)
        return out


class AtrousBasicBlock(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(AtrousBasicBlock,self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(inplanes,64, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=3,dilation=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,outplanes,kernel_size=3,stride=1,padding=5,dilation=5)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(inplanes,outplanes,kernel_size=1,stride=1,padding=0)
        self.bn4 = nn.BatchNorm2d(outplanes)

    def forward(self,x):
        identify = x
        identify = self.conv4(identify)
        identify = self.bn4(identify)
        identify = self.relu(identify)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)

        out += identify
        out = self.relu(out)
        return out




def upsample(in_channels, out_channels, mode='transpose'):
        if mode == 'transpose':
            return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
            )
        elif mode == 'bilinear':
            return nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                # conv1x1(in_channels, out_channels)
                )
        else:
            return nn.Sequential(
                nn.Upsample(mode='nearest', scale_factor=2),
                # conv1x1(in_channels, out_channels)
                )


class _upsample_add(nn.Module):
    def __init__(self,in_channels, out_channels, mode='transpose'):
        super(_upsample_add,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode
        self.upsample = upsample(self.in_channels, self.out_channels, self.mode)

    def forward(self,high,low):
        h = self.upsample(high)
        l = low
        out = h + l
        return out

