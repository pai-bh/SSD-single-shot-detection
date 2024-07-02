# model/ssd/default_boxes.py
import torch


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


class DefaultBoxGenerator:
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, scale_xy, scale_wh):
        """
        DefaultBoxGenerator 초기화

        Args:
            fig_size (int): 원본 이미지 크기
            feat_size (list): 각 feature map의 크기 리스트
            steps (list): 각 feature map의 셀 간 거리 리스트
            scales (list): 각 feature map의 기본 스케일 리스트
            aspect_ratios (list): 각 feature map의 종횡비 리스트
            scale_xy (float): 중심 좌표의 스케일링 요소
            scale_wh (float): 너비와 높이의 스케일링 요소
        """
        self.fig_size = fig_size
        self.feat_size = feat_size
        self.steps = steps
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.scale_xy = scale_xy
        self.scale_wh = scale_wh

    def __call__(self):
        """
        디폴트 박스(anchor) 생성

        Returns:
            torch.Tensor: 각 feature map에 대한 디폴트 박스 텐서
        """
        anchors = []

        for idx, sfeat in enumerate(self.feat_size):
            step = self.steps[idx]
            scale = self.scales[idx]

            for i in range(sfeat):
                for j in range(sfeat):
                    cx = (j + 0.5) * step / self.fig_size
                    cy = (i + 0.5) * step / self.fig_size

                    for ratio in self.aspect_ratios[idx]:
                        w = scale * torch.sqrt(torch.tensor(ratio))
                        h = scale / torch.sqrt(torch.tensor(ratio))
                        anchors.append([cx, cy, w, h])

                    anchors.append([cx, cy, scale, scale])
                    scale_next = self.scales[idx + 1] if idx + 1 < len(self.scales) else 1.0
                    anchors.append([cx, cy, torch.sqrt(torch.tensor(scale * scale_next)),
                                    torch.sqrt(torch.tensor(scale * scale_next))])

        anchors = torch.tensor(anchors, dtype=torch.float32)
        anchors.clamp_(0, 1)
        return anchors

    def encode(self, boxes, labels, iou_threshold=0.5):
        """
        디폴트 박스의 오프셋과 레이블을 계산

        Args:
            boxes (torch.Tensor): 실제 박스 좌표
            labels (torch.Tensor): 실제 레이블
            iou_threshold (float): IoU 임계값

        Returns:
            tuple: 오프셋과 레이블 텐서
        """
        default_boxes = self.__call__()
        ious = iou(default_boxes.unsqueeze(1), boxes.unsqueeze(0))
        best_iou, best_idx = ious.max(1)

        loc_targets = boxes[best_idx]
        loc_targets = torch.cat([
            (loc_targets[:, :2] + loc_targets[:, 2:]) / 2 - default_boxes[:, :2],  # 중심 좌표 오프셋
            (loc_targets[:, 2:] - loc_targets[:, :2]) / default_boxes[:, 2:]  # 너비와 높이 오프셋
        ], 1)
        loc_targets[:, :2] /= self.scale_xy
        loc_targets[:, 2:] /= self.scale_wh

        conf_targets = 1 + labels[best_idx]
        conf_targets[best_iou < iou_threshold] = 0

        return loc_targets, conf_targets

    def decode(self, loc_preds):
        """
        예측된 위치 오프셋을 실제 좌표로 변환

        Args:
            loc_preds (torch.Tensor): 위치 오프셋 예측값

        Returns:
            torch.Tensor: 실제 좌표로 변환된 박스들
        """
        default_boxes = self.__call__()

        boxes = torch.cat([
            default_boxes[:, :2] + loc_preds[:, :2] * self.scale_xy * default_boxes[:, 2:],
            default_boxes[:, 2:] * torch.exp(loc_preds[:, 2:] * self.scale_wh)
        ], 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        return boxes
