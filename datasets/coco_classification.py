import json
import os
from PIL import Image
from torch.utils.data import Dataset


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
        for anno in self.annotations['annotations']:
            if anno['image_id'] == img_info['id']:
                annotations.append(anno)

        if len(annotations) > 0:
            areas = [anno['area'] for anno in annotations]
            max_area_idx = areas.index(max(areas))
            category_id = annotations[max_area_idx]['category_id']
            category_id -= 1
        else:
            category_id = -1

        return image, category_id
