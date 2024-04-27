import os
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from core.CenterNet.center_net import CenterNet

from datasets.coco_object_detection import COCODetectionDataset

from configs.training_config import config as training_config
from configs.mobilenet_config import config as mobilenet_config
from configs.centernet_config import config as centernet_config
from torch.utils.data.dataloader import default_collate
from loss.centernet_loss import CenterNetLoss

from tqdm import tqdm


def custom_collate(batch):
    max_len = max(len(item['sizes']) for _, item in batch)
    for _, item in batch:
        pad_len = max_len - len(item['sizes'])
        item['sizes'] = torch.cat([item['sizes'], torch.zeros(pad_len, 2)], dim=0)
        item['offsets'] = torch.cat([item['offsets'], torch.zeros(pad_len, 2)], dim=0)
        item['size_mask'] = torch.cat([item['size_mask'], torch.zeros(pad_len)], dim=0)
        item['offset_mask'] = torch.cat([item['offset_mask'], torch.zeros(pad_len)], dim=0)
    return default_collate(batch)


##############################################################################################
# 현재 설정 출력
print(f"모바일넷 백본 경량화 선택: {mobilenet_config.BACKBONE.LIGHTWEIGHT}")
print(f"학습 구분 선택: {training_config.TYPE.TRAIN_TYPE}")
print(f"train 데이터셋 경로: {training_config.DATASET.TRAIN_PATH}")
print(f"valid 데이터셋 경로: {training_config.DATASET.VALID_PATH}")
print(f"epoch 선택: {training_config.TRAIN.EPOCH}")
print(f"learning rate 선택: {training_config.TRAIN.LR}")

##############################################################################################
# 데이터 전처리
resize = training_config.TRANSFORM.RESIZE
mean = training_config.TRANSFORM.MEAN
std = training_config.TRANSFORM.STD
transform = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[std])
])

##############################################################################################
# 데이터셋 경로
annotation_file_name = training_config.DATASET.ANNOTATION

train_dir = training_config.DATASET.TRAIN_PATH
train_annotation_path = os.path.join(train_dir, annotation_file_name)

val_dir = training_config.DATASET.VALID_PATH
val_annotation_path = os.path.join(val_dir, annotation_file_name)

# 데이터 로더 생성
train_dataset = COCODetectionDataset(root_dir=train_dir,
                                     annotation_file=train_annotation_path,
                                     transform=transform)
val_dataset = COCODetectionDataset(root_dir=val_dir,
                                   annotation_file=val_annotation_path,
                                   transform=transform)

train_loader = DataLoader(train_dataset, batch_size=training_config.TRAIN.BATCH_SIZE, shuffle=True,
                          collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=training_config.TRAIN.BATCH_SIZE)

##############################################################################################
# 모델 생성
num_classes = training_config.DATASET.NUM_CLASSES
model = CenterNet(num_classes=num_classes)

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"\nUsing device: {device}")

if torch.cuda.is_available():
    print("GPU is available.")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    print(f"GPU device count: {torch.cuda.device_count()}")
else:
    print("GPU is not available. Using CPU instead.")

# 손실 함수와 옵티마이저 정의
criterion = CenterNetLoss()
optimizer = optim.Adam(model.parameters(), lr=training_config.TRAIN.LR)

##############################################################################################
# 학습 루프
num_epochs = training_config.TRAIN.EPOCH

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", unit="batch")
    for images, targets in progress_bar:
        images = images.to(device)
        target_heatmap = targets['heatmap'].to(device)
        target_sizes = targets['sizes'].to(device)
        target_offsets = targets['offsets'].to(device)
        target_size_mask = targets['size_mask'].to(device)
        target_offset_mask = targets['offset_mask'].to(device)

        # 모델 예측
        predictions = model(images)

        # 손실 계산
        loss = criterion(predictions,
                         (target_heatmap, target_sizes, target_offsets, target_size_mask, target_offset_mask))

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 손실 업데이트
        train_loss += loss.item()

        # 진행 상황 업데이트
        progress_bar.set_postfix(loss=loss.item())

        # 평균 손실 계산
    train_loss /= len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")

    ##############################################################################################
    # 검증 루프 (선택적)
    if val_loader:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                target_heatmap = targets['heatmap'].to(device)
                target_sizes = targets['sizes'].to(device)
                target_offsets = targets['offsets'].to(device)
                target_size_mask = targets['size_mask'].to(device)
                target_offset_mask = targets['offset_mask'].to(device)

                # 모델 예측
                predictions = model(images)

                # 손실 계산
                loss = criterion(predictions,
                                 (target_heatmap, target_sizes, target_offsets, target_size_mask, target_offset_mask))

                val_loss += loss.item()

        # 평균 검증 손실 출력
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

##############################################################################################
# 학습이 완료된 후 모델 가중치 저장
torch.save(model.state_dict(), training_config.WEIGHT.SAVE_PATH)
print("\nModel weights saved successfully.")
