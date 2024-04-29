import torch
import torch.nn as nn
from configs.mobilenet_config import config as mobilenet_config

from core.MobileNet.mobile_net import MobileNetV1BackboneGray


class CenterNetHead(nn.Module):
    def __init__(self, in_channels, num_classes=12):
        super(CenterNetHead, self).__init__()
        # 각 클래스의 객체 중심을 예측하기 위해 사용됩니다. 모든 가능한 객체 위치에 대해 해당 클래스의 객체 중심이 있을 확률을 출력합니다. nn.Conv2d를 사용하여 입력 채널에서 클래스 수만큼의
        # 출력 채널을 갖는 히트맵을 생성합니다.
        self.heatmap = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True)
        # 히트맵에서 예측된 중심점이 실제 중심에서 약간 벗어날 수 있기 때문에, 이를 미세 조정하는 데 사용됩니다. 이는 각 중심점 위치의 x, y 좌표에 대한 보정값을 제공합니다.
        self.offset = nn.Conv2d(in_channels, 2, kernel_size=1, bias=True)
        # 각 객체의 실제 너비와 높이를 예측합니다. 이 값은 각 객체에 대한 바운딩 박스를 생성하는 데 필요합니다.
        self.size = nn.Conv2d(in_channels, 2, kernel_size=1, bias=True)

    def forward(self, x):
        heatmap = self.heatmap(x)
        offset = self.offset(x)
        size = self.size(x)
        return heatmap, offset, size


class CenterNet(nn.Module):
    def __init__(self, num_classes=12):
        super(CenterNet, self).__init__()
        self.backbone = MobileNetV1BackboneGray(num_classes)

        if mobilenet_config.BACKBONE.LIGHTWEIGHT:
            self.head = CenterNetHead(512, num_classes)
        else:
            self.head = CenterNetHead(1024, num_classes)

        # 히트맵 업샘플링 레이어 추가
        self.heatmap_upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        features = self.backbone.model(x)
        heatmap, offset, size = self.head(features)

        # 히트맵, 오프셋, 크기 증가
        heatmap = self.heatmap_upsample(heatmap)

        return heatmap, offset, size
