import torch
import torch.nn as nn
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from configs.mobilenet_config import config as mobilenet_config


# 모바일넷 V1 백본 수정
class MobileNetV1BackboneGray(nn.Module):
    def __init__(self, num_classes=12):
        super(MobileNetV1BackboneGray, self).__init__()

        # lightweight: 경량화 모델을 쓸건지
        if mobilenet_config.BACKBONE.LIGHTWEIGHT:
            self.model = self.backbone_v1_1()
        else:
            self.model = self.backbone_v1_0()

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def conv_bn(self, inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )

    def conv_dw(self, inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def backbone_v1_0(self):
        model = nn.Sequential(
            self.conv_bn(1, 32, 2),
            self.conv_dw(32, 64, 1),
            self.conv_dw(64, 128, 2),
            self.conv_dw(128, 128, 1),
            self.conv_dw(128, 256, 2),
            self.conv_dw(256, 256, 1),
            self.conv_dw(256, 512, 2),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 1024, 2),
            self.conv_dw(1024, 1024, 1),
        )

        return model

    def backbone_v1_1(self):
        model = nn.Sequential(
            self.conv_bn(1, 16, 2),
            self.conv_dw(16, 32, 1),
            self.conv_dw(32, 64, 2),
            self.conv_dw(64, 64, 1),
            self.conv_dw(64, 128, 2),
            self.conv_dw(128, 128, 1),
            self.conv_dw(128, 256, 2),
            self.conv_dw(256, 256, 1),
            self.conv_dw(256, 256, 1),
            self.conv_dw(256, 256, 1),
            self.conv_dw(256, 256, 1),
            self.conv_dw(256, 256, 1),
            self.conv_dw(256, 512, 2),
            self.conv_dw(512, 512, 1),
        )

        return model
