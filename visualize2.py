# visualize_tmp.py
import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
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
def get_parameters(feature_map_size):
    if feature_map_size == 38:
        step = 8
        scale = 0.1
        scale_next = 0.2
        aspect_ratios = [1.0, 2.0, 0.5]
    elif feature_map_size == 19:
        step = 16
        scale = 0.2
        scale_next = 0.375
        aspect_ratios = [1.0, 2.0, 0.5, 3.0, 1 / 3]
    elif feature_map_size == 10:
        step = 32
        scale = 0.375
        scale_next = 0.55
        aspect_ratios = [1.0, 2.0, 0.5, 3.0, 1 / 3]
    elif feature_map_size == 5:
        step = 64
        scale = 0.55
        scale_next = 0.725
        aspect_ratios = [1.0, 2.0, 0.5, 3.0, 1 / 3]
    elif feature_map_size == 3:
        step = 100
        scale = 0.725
        scale_next = 0.9
        aspect_ratios = [1.0, 2.0, 0.5]
    elif feature_map_size == 1:
        step = 300
        scale = 0.9
        scale_next = 1.0
        aspect_ratios = [1.0, 2.0, 0.5]
    else:
        raise ValueError("Unsupported feature map size")
    return step, scale, scale_next, aspect_ratios


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

    # 이미지 영역을 초과하는 경우 클리핑
    clipped_boxes = []
    for box in default_boxes:
        cx, cy, w, h = box
        xmin = max(cx - w / 2, 0)
        ymin = max(cy - h / 2, 0)
        xmax = min(cx + w / 2, 300)
        ymax = min(cy + h / 2, 300)
        clipped_boxes.append((xmin, ymin, xmax - xmin, ymax - ymin))

    return clipped_boxes


def create_animation(feature_map_size):
    step, scale, scale_next, aspect_ratios = get_parameters(feature_map_size)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_np)
    ax.set_title(f'Feature Map {feature_map_size}x{feature_map_size}')
    ax.grid(True)

    # 모든 센터 포인트 시각화
    for i in range(feature_map_size):
        for j in range(feature_map_size):
            cx = (j + 0.5) * step
            cy = (i + 0.5) * step
            ax.plot(cx, cy, 'ro', markersize=2)

    # 초기화 함수
    def init():
        # 어노테이션 박스 시각화
        annotation_patches = []
        for bbox in adjusted_bboxes:
            xmin, ymin, xmax, ymax = bbox
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='b',
                                     facecolor='none')
            ax.add_patch(rect)
            annotation_patches.append(rect)
        return annotation_patches

    default_box_patches = []
    center_point = []

    # 업데이트 함수
    def update(frame):
        nonlocal default_box_patches
        nonlocal center_point

        # 이전 패치 제거
        for patch in default_box_patches:
            patch.remove()
        default_box_patches = []

        for point in center_point:
            point.remove()
        center_point = []

        i, j = divmod(frame, feature_map_size)
        cx = (j + 0.5) * step
        cy = (i + 0.5) * step

        # 센터 포인트 시각화
        point = ax.plot(cx, cy, 'ro', markersize=4)
        center_point.extend(point)

        # 디폴트 박스 시각화
        default_boxes = generate_default_boxes(cx, cy, scale, aspect_ratios, scale_next)
        for box in default_boxes:
            xmin, ymin, w, h = box
            rect = patches.Rectangle((xmin, ymin), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            default_box_patches.append(rect)

        # 어노테이션 박스 시각화
        annotation_patches = []
        for bbox in adjusted_bboxes:
            xmin, ymin, xmax, ymax = bbox
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='b',
                                     facecolor='none')
            ax.add_patch(rect)
            annotation_patches.append(rect)

        return annotation_patches + default_box_patches + center_point

    # 애니메이션 생성 및 GIF 저장
    ani = FuncAnimation(fig, update, frames=range(feature_map_size * feature_map_size), init_func=init, blit=True,
                        interval=50)
    ani.save(f"default_boxes_{feature_map_size}x{feature_map_size}.gif", writer=PillowWriter(fps=20))
    plt.show()


# 원하는 feature_map_size로 애니메이션 생성
create_animation(10)  # 예시: 19x19 feature map
