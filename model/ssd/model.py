# model/ssd/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from base.base_model import BaseModel
from .default_boxes import DefaultBoxGenerator


class SSD(BaseModel):
    """
    SSD 모델 클래스
    Single Shot MultiBox Detector 모델을 구현합니다.
    """

    def __init__(self, num_classes):
        """
        SSD 모델 초기화

        Args:
            num_classes (int): 분류할 클래스의 수
        """
        super(SSD, self).__init__()
        self.num_classes = num_classes

        # VGG16 특징 추출기 초기화 (미리 학습된 가중치 사용)
        self.vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features

        # 추가 컨볼루션 레이어 정의
        self.extras = self._add_extras([512, 256, 512, 128, 256, 128, 256, 128, 256, 128], 1024)

        # 위치 예측 레이어 정의
        self.loc = nn.ModuleList([
            nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),  # conv4_3 layer feature map
            nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),  # conv7 (fc7) layer feature map
            nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),  # 추가 컨볼루션 layer feature map
            nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),  # 추가 컨볼루션 layer feature map
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),  # 추가 컨볼루션 layer feature map
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)  # 추가 컨볼루션 layer feature map
        ])

        # 분류 예측 레이어 정의
        self.conf = nn.ModuleList([
            nn.Conv2d(512, 4 * self.num_classes, kernel_size=3, padding=1),  # conv4_3 layer feature map
            nn.Conv2d(1024, 6 * self.num_classes, kernel_size=3, padding=1),  # conv7 (fc7) layer feature map
            nn.Conv2d(512, 6 * self.num_classes, kernel_size=3, padding=1),  # 추가 컨볼루션 layer feature map
            nn.Conv2d(256, 6 * self.num_classes, kernel_size=3, padding=1),  # 추가 컨볼루션 layer feature map
            nn.Conv2d(256, 4 * self.num_classes, kernel_size=3, padding=1),  # 추가 컨볼루션 layer feature map
            nn.Conv2d(256, 4 * self.num_classes, kernel_size=3, padding=1)  # 추가 컨볼루션 layer feature map
        ])

        # Default Box Generator 초기화
        self.default_box_generator = DefaultBoxGenerator(
            fig_size=300,  # 원본 이미지 크기
            feat_size=[38, 19, 10, 5, 3, 1],  # 각 feature map의 크기 리스트
            steps=[8, 16, 32, 64, 100, 300],  # 각 feature map의 셀 간 거리 리스트
            scales=[0.1, 0.2, 0.375, 0.55, 0.725, 0.9],  # 각 feature map의 기본 스케일 리스트
            aspect_ratios=[
                [1.0, 2.0, 0.5],  # 첫 번째 feature map에 대해 종횡비 1:1, 2:1, 1:2 사용
                [1.0, 2.0, 3.0, 0.5, 1 / 3],  # 두 번째 feature map에 대해 종횡비 1:1, 2:1, 3:1, 1:2, 1:3 사용
                [1.0, 2.0, 3.0, 0.5, 1 / 3],  # 세 번째 feature map에 대해 동일한 종횡비 사용
                [1.0, 2.0, 3.0, 0.5, 1 / 3],  # 네 번째 feature map에 대해 동일한 종횡비 사용
                [1.0, 2.0, 3.0, 0.5, 1 / 3],  # 다섯 번째 feature map에 대해 동일한 종횡비 사용
                [1.0, 2.0, 0.5]  # 여섯 번째 feature map에 대해 종횡비 1:1, 2:1, 1:2 사용
            ],
            scale_xy=0.1,  # 중심 좌표의 스케일링 요소
            scale_wh=0.2  # 너비와 높이의 스케일링 요소
        )
        self.default_boxes = self.default_box_generator()

    def _add_extras(self, cfg, in_channels):
        """
        추가 컨볼루션 레이어 정의

        Args:
            cfg (list): 레이어 구성 설정
            in_channels (int): 입력 채널 수

        Returns:
            nn.ModuleList: 추가 레이어 모듈 리스트
        """
        layers = []
        for k, v in enumerate(cfg):
            if k % 2 == 0:
                layers += [nn.Conv2d(in_channels, v, kernel_size=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, stride=2, padding=1)]
            in_channels = v
        return nn.ModuleList(layers)

    def forward(self, x):
        """
        모델의 순전파 정의

        Args:
            x (torch.Tensor): 입력 데이터

        Returns:
            tuple: 위치 예측값(loc_preds)과 분류 예측값(conf_preds)
        """
        features = list()
        locs = list()
        confs = list()

        # VGG16의 초기 계층(첫 23개 계층)을 통과하여 특징 맵을 생성
        for i in range(23):
            x = self.vgg[i](x)
        features.append(x)  # 첫 번째 피쳐 맵 (conv4_3 레이어의 출력)

        # VGG16의 나머지 계층(24번째부터 마지막 계층까지)을 통과하여 또 다른 피쳐 맵을 생성
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        features.append(x)  # 두 번째 피쳐 맵 (conv7 또는 fc7 레이어의 출력)

        # 추가 컨볼루션 레이어를 통과하여 나머지 피쳐 맵을 생성
        for i in range(len(self.extras)):
            x = F.relu(self.extras[i](x), inplace=True)
            if i % 2 == 1:
                features.append(x)  # 나머지 4개의 피쳐 맵

        # 위치 및 분류 예측
        for i, feature in enumerate(features):
            locs.append(self.loc[i](feature).permute(0, 2, 3, 1).contiguous())
            confs.append(self.conf[i](feature).permute(0, 2, 3, 1).contiguous())

        # 위치 예측값과 분류 예측값을 연결하여 최종 예측값 생성
        loc_preds = torch.cat([o.view(o.size(0), -1) for o in locs], 1)
        conf_preds = torch.cat([o.view(o.size(0), -1) for o in confs], 1)

        # 위치 예측값을 [batch_size, num_boxes, 4] 형태로 변환
        loc_preds = loc_preds.view(loc_preds.size(0), -1, 4)

        # 분류 예측값을 [batch_size, num_boxes, num_classes] 형태로 변환
        conf_preds = conf_preds.view(conf_preds.size(0), -1, self.num_classes)

        return loc_preds, conf_preds

    def detect(self, loc_preds, conf_preds, threshold=0.5):
        """
        예측된 위치(loc_preds)와 분류(conf_preds)를 기반으로 최종 객체를 탐지하는 기능

        Args:
            loc_preds (torch.Tensor): 위치 예측값 (디폴트 박스에 대한 오프셋)
            conf_preds (torch.Tensor): 분류 예측값 (각 클래스에 대한 신뢰도 점수)
            threshold (float): NMS 임계값 (디폴트: 0.5)

        Returns:
            tuple: 최종 탐지 결과 (박스와 점수)
        """
        boxes = []  # 최종 탐지된 박스를 저장할 리스트
        scores = []  # 각 박스의 점수를 저장할 리스트

        # 디폴트 박스를 실제 좌표로 변환
        default_boxes = self.default_box_generator.decode(loc_preds)

        for i in range(loc_preds.size(1)):
            box = default_boxes[:, i, :]
            score = conf_preds[:, i, :]

            for j in range(1, self.num_classes):
                # 해당 클래스에 대한 신뢰도 점수가 임계값보다 큰 박스만 선택
                mask = score[:, j] > threshold
                if mask.sum() == 0:
                    continue

                # 선택된 박스와 해당 점수를 추출
                box_mask = box[mask, :]
                score_mask = score[mask, j]

                # Non-Maximum Suppression (NMS) 적용
                keep = nms(boxes=box_mask, scores=score_mask, iou_threshold=threshold)

                # NMS를 통과한 박스와 점수를 저장
                boxes.append(box_mask[keep])
                scores.append(score_mask[keep])

        return boxes, scores


def nms(boxes, scores, iou_threshold):
    """
    박스들의 교집합 비율(IoU)에 따라 비최대 억제(NMS)를 수행합니다.

    Args:
        boxes (torch.Tensor): 예측된 박스들 (tensor of shape [num_boxes, 4])
        scores (torch.Tensor): 박스의 신뢰도 점수 (tensor of shape [num_boxes])
        iou_threshold (float): IoU 임계값 (float)

    Returns:
        torch.Tensor: NMS를 통과한 박스들의 인덱스 (tensor of shape [num_keep_boxes])
    """
    keep = []  # 최종적으로 유지할 박스들의 인덱스를 저장할 리스트
    _, indices = scores.sort(descending=True)  # 점수에 따라 박스들을 내림차순으로 정렬한 인덱스

    while indices.numel() > 0:
        max_score_index = indices[0]  # 가장 높은 점수를 가진 박스의 인덱스
        keep.append(max_score_index.item())  # 해당 인덱스를 유지할 리스트에 추가

        if indices.numel() == 1:
            break  # 남은 박스가 하나일 경우 루프 종료

        # 현재 가장 높은 점수를 가진 박스와 나머지 박스들 간의 IoU 계산
        ious = iou(boxes[max_score_index].unsqueeze(0), boxes[indices[1:]])

        # IoU가 임계값 이하인 박스들만 남기고 나머지 제거
        indices = indices[1:][ious <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long)  # 유지할 박스들의 인덱스를 텐서로 반환


def iou(box1, box2):
    """
    두 박스 간의 IoU 계산 함수

    Args:
        box1 (torch.Tensor): 첫 번째 박스
        box2 (torch.Tensor): 두 번째 박스들

    Returns:
        torch.Tensor: IoU 값
    """
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])

    inter_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area
