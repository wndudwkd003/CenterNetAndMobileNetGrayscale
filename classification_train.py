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

from tqdm import tqdm


##############################################################################################
# 데이터 전처리
resize = training_config.TRANSFORM.RESIZE
mean = training_config.TRANSFORM.RESIZE
std = training_config.TRANSFORM.RESIZE
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
train_dataset = COCOClassificationDataset(root_dir=train_dir,
                                          annotation_file=train_annotation_path,
                                          transform=transform)
val_dataset = COCOClassificationDataset(root_dir=val_dir,
                                        annotation_file=val_annotation_path,
                                        transform=transform)

train_loader = DataLoader(train_dataset, batch_size=training_config.TRAIN.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=training_config.TRAIN.BATCH_SIZE)


##############################################################################################
# 모델 생성
num_classes = training_config.DATASET.NUM_CLASSES
model = MobileNetV1BackboneGray(num_classes=num_classes)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=training_config.TRAIN.LR)

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"Using device: {device}")

if torch.cuda.is_available():
    print("GPU is available.")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    print(f"GPU device count: {torch.cuda.device_count()}")
else:
    print("GPU is not available. Using CPU instead.")


##############################################################################################
# 학습 루프
num_epochs = training_config.TRAIN.EPOCH

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", unit="batch")
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 진행 바 업데이트
        progress_bar.set_postfix({"Loss": loss.item()})

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}")

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
        print(f"\nValidation Accuracy: {accuracy:.2f}%")


##############################################################################################
# 학습이 완료된 후 모델 가중치 저장
torch.save(model.state_dict(), 'models/classification/model_weights.pth')
print("\nModel weights saved successfully.")