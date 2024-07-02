# visualize_tmp.py
import os
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


# IoU 계산 함수
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union


# 이미지 경로 및 어노테이션 경로
image_file_name = '2011_004420.jpg'
annotation_file_name = '2011_004420.xml'
image_folder_path = 'train/data/VOCdevkit/VOC2012/JPEGImages/'
annotation_folder_path = 'train/data/VOCdevkit/VOC2012/Annotations/'

image_path = os.path.join(image_folder_path, image_file_name)
annotation_path = os.path.join(annotation_folder_path, annotation_file_name)

print(f"Loaded image: {image_path}")
print(f"Loaded annotation: {annotation_path}")

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

# 센터 및 디폴트 박스 정의
feature_map_sizes = [38, 19, 10, 5, 3, 1]
steps = [8, 16, 32, 64, 100, 300]
scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
aspect_ratios = [
    [1.0, 2.0, 3.0, 0.5, 1 / 3],
    [1.0, 2.0, 3.0, 0.5, 1 / 3],
    [1.0, 2.0, 3.0, 0.5, 1 / 3],
    [1.0, 2.0, 3.0, 0.5, 1 / 3],
    [1.0, 2.0, 0.5],
    [1.0, 2.0, 0.5]
]


def generate_default_boxes(cx, cy, scale, aspect_ratios, scale_next):
    default_boxes = []

    # 기본 박스
    default_boxes.append((cx, cy, scale * 300, scale * 300))
    default_boxes.append((cx, cy, torch.sqrt(torch.tensor(scale * scale_next)) * 300,
                          torch.sqrt(torch.tensor(scale * scale_next)) * 300))

    for ar in aspect_ratios:
        default_boxes.append(
            (cx, cy, scale * torch.sqrt(torch.tensor(ar)) * 300, scale / torch.sqrt(torch.tensor(ar)) * 300))
        default_boxes.append(
            (cx, cy, scale / torch.sqrt(torch.tensor(ar)) * 300, scale * torch.sqrt(torch.tensor(ar)) * 300))

    return default_boxes


# 시각화
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for idx, (feature_map_size, step, scale, aspect_ratio) in enumerate(
        zip(feature_map_sizes, steps, scales, aspect_ratios)):
    scale_next = scales[idx + 1] if idx + 1 < len(scales) else 1.0

    ax = axes[idx // 3, idx % 3]
    ax.imshow(image_np)
    ax.set_title(f'Feature Map {feature_map_size}x{feature_map_size}')

    for i in range(feature_map_size):
        for j in range(feature_map_size):
            cx = (j + 0.5) * step
            cy = (i + 0.5) * step

            # 특정 center에 대한 디폴트 박스 생성
            default_boxes = generate_default_boxes(cx, cy, scale, aspect_ratio, scale_next)
            for box in default_boxes:
                cx, cy, w, h = box
                default_bbox = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
                for gt_bbox in adjusted_bboxes:
                    iou = calculate_iou(default_bbox, gt_bbox)
                    if iou >= 0.5:
                        rect = patches.Rectangle((cx - w / 2, cy - h / 2), w, h, linewidth=1, edgecolor='r',
                                                 facecolor='none')
                        ax.add_patch(rect)

    # 어노테이션 박스 시각화
    for bbox in adjusted_bboxes:
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

plt.tight_layout()
plt.show()
