# dataprovider/data_setter.py
from base.base_data_setter import BaseDataSetter
from torchvision.datasets import CIFAR10
from torchvision.datasets import VOCDetection
import torch


class CIFAR10DataSetter(BaseDataSetter):
    def __init__(self, *args, **kwargs):
        self.cifar10 = CIFAR10(*args, **kwargs)

    def __getitem__(self, item):
        return self.cifar10.__getitem__(item)

    def __len__(self):
        return len(self.cifar10)


class VOCDataSetter(BaseDataSetter):
    """
    VOC 데이터셋을 위한 DataSetter 클래스.
    BaseDataSetter를 상속받습니다.
    """

    def __init__(self, root, year, image_set, download, transform=None, target_transform=None):
        """
        VOC 데이터셋 초기화
        """
        self.voc = VOCDetection(root=root, year=year, image_set=image_set, download=download)
        self.transform = transform
        self.target_transform = target_transform
        # VOCDetection 객체 초기화 및 변환 설정

    def __getitem__(self, item):
        """
        데이터셋의 특정 항목을 반환
        """
        image, target = self.voc.__getitem__(item)
        # 이미지와 타겟을 반환합니다.
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        # target을 tensor 형식으로 변환
        boxes = []
        labels = []
        for obj in target['annotation']['object']:
            bbox = obj['bndbox']
            boxes.append([
                float(bbox['xmin']),
                float(bbox['ymin']),
                float(bbox['xmax']),
                float(bbox['ymax'])
            ])
            labels.append(1)  # 여기에 VOC 데이터셋의 라벨 인덱스를 추가하세요.

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        return image, {'boxes': boxes, 'labels': labels}

    def __len__(self):
        """
        데이터셋의 길이를 반환
        """
        return len(self.voc)
        # 데이터셋의 총 항목 수 반환
