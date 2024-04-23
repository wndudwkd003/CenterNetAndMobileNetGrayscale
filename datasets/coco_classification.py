import json
import os
from PIL import Image
from torch.utils.data import Dataset

from configs.training_config import config as training_config


class COCOClassificationDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_info = self.annotations['images']
        self.category_info = {cat['id']: cat for cat in self.annotations['categories']}
        # print(f"image_info: {self.image_info}")
        # print(f"image_info: {self.category_info}")

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        img_info = self.image_info[idx]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        # image = Image.open(img_path).convert('RGB')
        image = Image.open(img_path)  # 원본 채널 그대로 오픈

        # print(f"Image mode: {image.mode}")  # 이미지의 모드 출력

        if self.transform:
            image = self.transform(image)

        annotations = []
        for annotation in self.annotations['annotations']:
            if annotation['image_id'] == img_info['id']:
                annotations.append(annotation)

        if len(annotations) == 0:
            if training_config.DEBUG.DEBUG_PRINT:
                print(f"\n어노테이션 없음 -> img_info: {img_info}, annotations: {annotations}")

            # 어노테이션이 없는 경우 다음 이미지로 이동
            idx = (idx + 1) % len(self)
            return self.__getitem__(idx)

        areas = [anno['area'] for anno in annotations]
        max_area_idx = areas.index(max(areas))
        category_id = annotations[max_area_idx]['category_id']

        return image, category_id
