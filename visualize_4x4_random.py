# visualize_4x4_random_colored_bboxes.py
import os
import random
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import xml.etree.ElementTree as ET


# 어노테이션 로드 함수
def load_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    size = root.find('size')
    original_width = int(size.find('width').text)
    original_height = int(size.find('height').text)

    bboxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bboxes.append((xmin, ymin, xmax, ymax))
    return bboxes, original_width, original_height


# 이미지 및 어노테이션 폴더 경로
image_folder_path = 'train/data/VOCdevkit/VOC2012/JPEGImages/'
annotation_folder_path = 'train/data/VOCdevkit/VOC2012/Annotations/'

# 이미지 및 어노테이션 파일 목록 가져오기
image_files = [f for f in os.listdir(image_folder_path) if f.endswith('.jpg')]
annotation_files = [f for f in os.listdir(annotation_folder_path) if f.endswith('.xml')]

# 파일명으로 매칭되는 16개의 랜덤 이미지 선택
random_files = random.sample(image_files, 16)

# 시각화
fig, axes = plt.subplots(4, 4, figsize=(12, 8))
colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

for idx, image_file_name in enumerate(random_files):
    annotation_file_name = image_file_name.replace('.jpg', '.xml')
    image_path = os.path.join(image_folder_path, image_file_name)
    annotation_path = os.path.join(annotation_folder_path, annotation_file_name)

    # 이미지 로드 및 변환
    image = Image.open(image_path)
    bboxes, original_width, original_height = load_annotation(annotation_path)
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image)
    image_np = image_tensor.permute(1, 2, 0).numpy()

    # Bounding box 좌표를 300x300 이미지 크기로 조정
    adjusted_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        xmin = xmin * 300 / original_width
        ymin = ymin * 300 / original_height
        xmax = xmax * 300 / original_width
        ymax = ymax * 300 / original_height
        adjusted_bboxes.append((xmin, ymin, xmax, ymax))

    # 각 이미지 표시
    ax = axes[idx // 4, idx % 4]
    ax.imshow(image_np)
    ax.set_title(f'VOC2012/{image_file_name}')

    # 어노테이션 박스 시각화
    for i, bbox in enumerate(adjusted_bboxes):
        color = colors[i % len(colors)]  # 여러 개의 bbox 색상을 다르게 설정
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    ax.axis('off')

plt.tight_layout()
plt.show()
