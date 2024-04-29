import json
import os
from PIL import Image
from torch.utils.data import Dataset

import torch
from configs.training_config import config as training_config
import numpy as np

from configs.centernet_config import config as centernet_config


class COCODetectionDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, output_stride=centernet_config.HYPER.STRIDE):
        self.root_dir = root_dir
        self.transform = transform
        self.output_stride = output_stride
        self.category_num = training_config.DATASET.NUM_CLASSES

        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_info = self.annotations['images']
        self.category_info = {cat['id']: cat for cat in self.annotations['categories']}
        self.max_annotations_num = max(len(anno) for anno in self.annotations['annotations'])
        # print(f"image_info: {self.image_info}")
        # print(f"image_info: {self.category_info}")

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        img_info = self.image_info[idx]
        img_file_name = img_info["file_name"]
        img_origin_h = img_info["height"]
        img_origin_w = img_info["width"]
        # print(f"\nfilename: {img_file_name}, h: {img_origin_h}, w: {img_origin_w}")
        img_path = os.path.join(self.root_dir, img_file_name)

        # image = Image.open(img_path).convert('RGB')
        image = Image.open(img_path)  # 원본 채널 그대로 오픈

        # print(f"Image mode: {image.mode}")  # 이미지의 모드 출력

        if self.transform:
            image = self.transform(image)

        image_c, image_h, image_w = image.shape

        annotations = [anno for anno in self.annotations['annotations'] if anno['image_id'] == img_info['id']]
        annotations_num = len(annotations)

        if annotations_num == 0:
            if training_config.DEBUG.DEBUG_PRINT:
                print(f"\n어노테이션 없음 -> img_info: {img_info}, annotations: {annotations}")

            # 어노테이션이 없는 경우 다음 이미지로 이동
            idx = (idx + 1) % len(self)
            return self.__getitem__(idx)

        # 히트맵은 각 객체 카테고리에 대해 별도의 채널을 가지고 있으며 각 채널은 해당 카테고리의 객체 중심을 예측함
        # 예를들어 히트맵의 첫 번째 채널이 Dog를 나타내면 이 채널에서 높은 값을 가지는 위치는 이미지에서 개의 중심이 있을 가능성이 높은 지점을 의미하는 것
        # 한마디로 각 픽셀 위치에서 해당 클래스의 객체 중심이 있을 확률을 나타내며 히트맵의 각 채널은 다른 객체 클래스를 대표
        heatmap = np.zeros((self.category_num, image_h // self.output_stride, image_w // self.output_stride),
                           dtype=np.float32)

        # 이미지 내의 각 객체(어노테이션)의 바운딩 박스 크기를 저장함. 정규화된 너비와 높이를 의미함
        sizes = np.zeros((annotations_num, 2), dtype=np.float32)
        offsets = np.zeros((2, image_h // self.output_stride, image_w // self.output_stride), dtype=np.float32)

        # 각 마스크는 유효한 크기와 오프셋 값이 있는지를 나타내며 손실 계산에서 사용됨. 이는 모델이 불필요한 위치에서 손실을 계산하는 것을 방지함
        # 각 바운딩 박스의 크기가 0보다 큰 경우 유효한 것으로 간주, 이는 각 객체의 크기 정보가 실제로 존재한다는 것을 의미함
        size_mask = np.zeros(annotations_num, dtype=np.float32)

        # 객체의 실제 중심 위치가 그리드 셀의 중심과 완전히 일치하지 않을 때, 오프셋 값은 유효하다. 이는 각 객체의 오프셋 정보가 실제로 존재한다는 것을 의미함
        offset_mask = np.zeros(annotations_num, dtype=np.float32)

        # print(f"idx: {idx}, annotations_num: {annotations_num}")

        for i, anno in enumerate(annotations):
            x, y, w, h = anno['bbox']  # 바운딩박스 x, y, w, h
            cx, cy = x + w / 2, y + h / 2  # 중앙 x, y좌표
            grid_x = int(cx // self.output_stride)
            grid_y = int(cy // self.output_stride)

            # 경계 검사
            grid_x = min(grid_x, image_w // self.output_stride - 1)  # 경계를 넘지 않도록 조정
            grid_y = min(grid_y, image_h // self.output_stride - 1)  # 경계를 넘지 않도록 조정

            heatmap[anno['category_id'] - 1, grid_y, grid_x] = 1
            sizes[i] = [w / image_w, w / image_h]
            offsets[0, grid_y, grid_x] = (cx - grid_x * self.output_stride) / self.output_stride
            offsets[1, grid_y, grid_x] = (cy - grid_y * self.output_stride) / self.output_stride
            size_mask[i] = 1 if w > 0 and h > 0 else 0
            offset_mask[i] = 1 if w > 0 and h > 0 else 0

            # print(f"w: {w}, h: {h}, image_w: {image_w}, image_h: {image_h}")

        # print(f"sizes: {sizes}")

        target = {
            'heatmap': torch.tensor(heatmap, dtype=torch.float32),
            'sizes': torch.tensor(sizes, dtype=torch.float32),
            'offsets': torch.tensor(offsets, dtype=torch.float32),
            'size_mask': torch.tensor(size_mask, dtype=torch.float32),
            'offset_mask': torch.tensor(offset_mask, dtype=torch.float32)
        }

        return image, target
