# model/ssd/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def hard_negative_mining(conf_loss, pos_mask, neg_pos_ratio):
    """
    Hard Negative Mining을 수행하는 함수

    Args:
        conf_loss (torch.Tensor): 각 샘플에 대한 손실 값
        pos_mask (torch.Tensor): 양성 샘플 마스크
        neg_pos_ratio (float): 음성 샘플과 양성 샘플의 비율

    Returns:
        torch.Tensor: 선택된 음성 샘플의 마스크
    """
    batch_size, num_boxes = conf_loss.shape
    conf_loss[pos_mask] = -float('inf')  # 양성 샘플의 손실 값을 무한대로 설정하여 선택되지 않도록 함

    _, idx = conf_loss.sort(dim=1, descending=True)  # 손실 값에 따라 정렬된 인덱스
    _, rank = idx.sort(dim=1)  # 정렬된 인덱스의 순위

    num_pos = pos_mask.sum(dim=1, keepdim=True)
    num_neg = neg_pos_ratio * num_pos

    neg_mask = rank < num_neg.expand_as(rank)
    return neg_mask


def ssd_loss(predictions, targets, num_classes, alpha=1.0, neg_pos_ratio=3.0):
    """
    SSD 손실 함수 계산

    Args:
        predictions (tuple): 모델의 예측값 (loc_preds, conf_preds)
        targets (tuple): 실제 값 (loc_targets, conf_targets)
        num_classes (int): 분류할 클래스의 수
        alpha (float): 위치 손실과 분류 손실 간의 균형을 맞추는 가중치
        neg_pos_ratio (float): Negative와 Positive 샘플의 비율

    Returns:
        torch.Tensor: 계산된 손실 값
    """
    loc_preds, conf_preds = predictions
    loc_targets, conf_targets = targets

    pos_mask = conf_targets > 0
    num_pos = pos_mask.sum(dim=1, keepdim=True)

    # 위치 손실 계산
    loc_loss = nn.SmoothL1Loss(reduction='none')(loc_preds, loc_targets)
    loc_loss = loc_loss.sum(dim=2)
    loc_loss = loc_loss[pos_mask].sum()

    # 분류 손실 계산
    conf_loss = F.cross_entropy(conf_preds.view(-1, num_classes), conf_targets.view(-1), reduction='none')
    conf_loss = conf_loss.view(conf_preds.size(0), -1)

    # Hard Negative Mining
    neg_mask = hard_negative_mining(conf_loss, pos_mask, neg_pos_ratio)
    conf_loss = conf_loss[pos_mask | neg_mask].sum()

    # 최종 손실 계산
    total_loss = loc_loss + alpha * conf_loss
    return total_loss / num_pos.sum()
