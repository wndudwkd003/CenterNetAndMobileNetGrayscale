# -*-coding:utf-8-*-

import numpy as np
import torch
import torch.nn as nn
import math

from configs.centernet_config import config as training_config


class CenterNetLoss(nn.Module):
    def __init__(self, alpha=2, beta=4, lambda_size=0.1, lambda_offset=1):
        super(CenterNetLoss, self).__init__()
        self.alpha = alpha # 이 매개변수는 포컬 로스(Focal Loss)에서 사용되며, 클래스 불균형 문제에 대응하기 위해 적용됩니다. alpha는 긍정 샘플(예: 객체 중심)의 중요성을 조정합니다. 보통 긍정 샘플의 수가 적을 때 더 큰 가중치를 주어 학습에 더 많은 영향을 끼치게 합니다.
        self.beta = beta # 이 역시 포컬 로스에서 사용되며, 쉬운 예제에 대한 모델의 관심을 줄이는 역할을 합니다. beta가 크면 쉬운 예제(모델이 이미 잘 예측하고 있는 경우)의 손실 기여도가 감소하며, 모델이 어려운 예제에 더 집중하도록 합니다.
        self.lambda_size = lambda_size # 바운딩 박스 크기 예측의 손실에 적용되는 가중치입니다. 이 값으로 손실의 전체적인 영향력을 조절하여, 히트맵 손실이나 오프셋 손실과의 균형을 맞춥니다.
        self.lambda_offset = lambda_offset # 오프셋 예측(객체 중심의 실제 위치와 그리드 셀 중심의 예상 위치 사이의 차이)의 손실에 적용되는 가중치입니다. 이 값은 오프셋 손실의 중요도를 조정하여, 다른 손실 요소들과 균형을 이루도록 합니다.
        self.epsilon = 1e-14 # 로그 함수의 입력값에 더함으로써, 로그 함수의 입력이 절대 0이 되지 않도록 하여 -∞ 발생을 방지

    # 클래스 불균형 문제를 해결
    def focal_loss(self, pred, target):
        pos_mask = target == 1
        neg_mask = target < 1

        print(f"focal_loss-> {len(pred)}, {len(target)}")
        print(f"pos_mask-> pos_mask: {pos_mask.shape}, neg_mask: {neg_mask.shape}")
        print(f"pos_mask-> pos_mask: {pos_mask.size()}, neg_mask: {neg_mask.size()}")
        pos_loss = -self.alpha * (1 - pred) ** self.beta * torch.log(pred + self.epsilon) * pos_mask
        neg_loss = -(1 - self.alpha) * pred ** self.beta * torch.log(1 - pred + self.epsilon) * neg_mask

        loss = pos_loss + neg_loss
        return loss.sum()

    # 히트맵 손실
    def heatmap_loss(self, pred_heatmap, target_heatmap):
        return self.focal_loss(pred_heatmap, target_heatmap)

    # 바운딩 박스 크기 손실 L1
    def size_loss(self, pred_sizes, target_sizes, mask):
        loss = torch.abs(pred_sizes - target_sizes) * mask
        return loss.sum() / (mask.sum() + self.epsilon)

    # 오프셋 손실 L1
    def offset_loss(self, pred_offsets, target_offsets, mask):
        loss = torch.abs(pred_offsets - target_offsets) * mask
        return loss.sum() / (mask.sum() + self.epsilon)

    # 최종 손실 함수는 위의 세가지 손실의 가중 합으로 계산
    def forward(self, predictions, targets):
        pred_heatmap, pred_offsets, pred_sizes = predictions
        target_heatmap, target_offsets, target_sizes, offset_mask, size_mask = targets

        print(f"pred_heatmap: {len(pred_heatmap)}, target_heatmap: {len(target_heatmap)}")
        print(f"pred_heatmap: {pred_heatmap.shape}, target_heatmap: {target_heatmap.shape}")

        hm_loss = self.heatmap_loss(pred_heatmap, target_heatmap)
        off_loss = self.offset_loss(pred_offsets, target_offsets, offset_mask)
        sz_loss = self.size_loss(pred_sizes, target_sizes, size_mask)

        total_loss = hm_loss + self.lambda_size * sz_loss + self.lambda_offset * off_loss
        return total_loss
