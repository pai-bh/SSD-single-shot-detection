#model/ssd/metric.py
import torch

def ssd_accuracy(predictions, targets, threshold=0.5):
    """
    SSD 모델의 정확도를 계산

    Args:
        predictions (tuple): 모델의 예측값 (loc_preds, conf_preds)
        targets (tuple): 실제 값 (loc_targets, conf_targets)
        threshold (float): 예측을 양성으로 간주할 신뢰도 임계값

    Returns:
        tuple: (정확도, 전체 샘플 수)
    """
    loc_preds, conf_preds = predictions
    loc_targets, conf_targets = targets

    conf_preds = torch.softmax(conf_preds, dim=-1)
    pos_mask = conf_targets > 0
    num_pos = pos_mask.sum()

    correct = (conf_preds.argmax(dim=-1) == conf_targets).float().sum()
    total = conf_targets.numel()

    accuracy = correct / total
    return accuracy.item(), total
