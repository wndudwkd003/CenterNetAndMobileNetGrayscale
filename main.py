import os
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from core.MobileNet.mobile_net import MobileNetV1BackboneGray

from datasets.coco_classification import COCOClassificationDataset

from configs.training_config import config as training_config

# 데이터 전처리
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 모델 생성
model = MobileNetV1BackboneGray(num_classes=12)

# 데이터셋 경로
annotation_file_name = training_config.DATASET.ANNOTATION

train_dir = training_config.DATASET.TRAIN_PATH
train_annotation_path = os.path.join(train_dir, annotation_file_name)

val_dir = training_config.DATASET.VALID_PATH
val_annotation_path = os.path.join(val_dir, annotation_file_name)

# 데이터 로더 생성
train_dataset = COCOClassificationDataset(root_dir=train_dir,
                                          annotation_file=train_annotation_path,
                                          transform=transform)
val_dataset = COCOClassificationDataset(root_dir=val_dir,
                                        annotation_file=val_annotation_path,
                                        transform=transform)

train_loader = DataLoader(train_dataset, batch_size=training_config.TRAIN.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=training_config.TRAIN.BATCH_SIZE)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=training_config.TRAIN.LR)

# 학습 루프
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:

        print(labels)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # 검증
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")
