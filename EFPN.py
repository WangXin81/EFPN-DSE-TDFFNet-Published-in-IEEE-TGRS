# the architecture of EFPN
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import models 
from torch.autograd import Variable
import numpy as np
from utils import _upsample_add, AtrousBasicBlock, BasicBlock



class new_resnet_fpn(nn.Module):
    def __init__(self,num_classes=21,pretrained=False, extract_features=False):
        super(new_resnet_fpn,self).__init__()
        self.extract_features = extract_features
        resnet = models.resnet34(pretrained=pretrained)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)
        self.avgpool = nn.Sequential(resnet.avgpool)

        # Top layer
        self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        
        # Lateral layers
        self.latlayer1 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 128, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 64, 256, kernel_size=1, stride=1, padding=0)
        
        # Smooth layers
        self.smooth0 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Deconv layers
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256,256,kernel_size=3,stride=2,padding=1,output_padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256,256,kernel_size=3,stride=2,padding=1,output_padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )


        # P4,P2 processing
        self.newlatlayer1 = nn.Conv2d(256,256,kernel_size=1,stride=1,padding=0)
        self.newlatlayer2 = nn.Conv2d(256,256,kernel_size=1,stride=1,padding=0)
        # P4,P2 continue processing
        self.basiclayer1 = BasicBlock(256,64)
        self.basiclayer2 = BasicBlock(64,256)
        self.basiclayer3 = AtrousBasicBlock(256,64)
        self.basiclayer4 = AtrousBasicBlock(64,256)

        #top_down
        self.topfc = nn.AdaptiveAvgPool2d((1, 1))
        self.downfc = nn.AdaptiveAvgPool2d((1, 1))
        
        #final
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        self._upsample_add1 = _upsample_add(256,256,mode='bilinear')
        self._upsample_add2 = _upsample_add(256,256,mode='bilinear')
        self._upsample_add3 = _upsample_add(256,256,mode='bilinear')

    def forward(self,x):
        # backbone
        x = self.layer0(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        #top-down pathway
        p5 = self.toplayer(c5)
        p4 = self._upsample_add1(p5,self.latlayer1(c4))
        p3 = self._upsample_add2(p4,self.latlayer2(c3))
        p2 = self._upsample_add3(p3,self.latlayer3(c2))

        #smooth
        p5 = self.smooth0(p5)
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        #DSE
        p5 = self.deconv1(p5)
        p4 = self.newlatlayer1(p4)
        p3 = self.deconv2(p3)
        p2 = self.newlatlayer2(p2)
        d4 = p5+p4
        d2 = p3+p2

        #TDFF
        d4_1 = self.basiclayer1(d4)
        d4_2 = self.basiclayer2(d4_1)
        d2_1 = self.basiclayer3(d2)
        d2_2 = self.basiclayer4(d2_1)

        x1 = self.topfc(d4_2)
        x2 = self.downfc(d2_2)

        x1 = x1.reshape(x1.size(0), -1)
        x2 = x2.reshape(x2.size(0), -1)
        x = torch.cat((x1,x2),1)

        #add
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x



# net = new_resnet_fpn(pretrained=False)
# net(Variable(torch.randn(24,3,256,256)))
