# train/train.py
from dataprovider.data_setter import VOCDataSetter
from dataprovider.data_loader import VOCDataLoader
from torch.utils.data import Subset
import torchvision.transforms as transforms
import torch.optim as optim
from model.ssd.loss import ssd_loss
from model.ssd.metric import ssd_accuracy
from model.ssd.model import SSD
from trainer import SSDTrainer
import torch


def collate_fn(batch):
    """
    데이터 로더를 위한 배치 구성 함수

    Args:
        batch (list): 배치 데이터

    Returns:
        tuple: 패딩된 이미지와 타겟
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    max_width = 300
    max_height = 300
    padded_images = []

    for image in images:
        padding = (0, 0, max_width - image.size(2), max_height - image.size(1))
        padded_image = transforms.functional.pad(image, padding, fill=0)
        padded_images.append(padded_image)

    padded_images = torch.stack(padded_images, dim=0)

    boxes = [target['boxes'] for target in targets]
    labels = [target['labels'] for target in targets]

    return padded_images, {'boxes': boxes, 'labels': labels}


if __name__ == '__main__':
    epochs = 1
    batch_size = 64

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])

    train_data_setter = VOCDataSetter(root='./data', year='2012', image_set='train', download=True, transform=transform)
    train_data_loader = VOCDataLoader(train_data_setter, batch_size=batch_size, shuffle=True, num_workers=4,
                                      collate_fn=collate_fn)

    valid_data_setter = VOCDataSetter(root='./data', year='2012', image_set='val', download=True, transform=transform)
    valid_data_loader = VOCDataLoader(valid_data_setter, batch_size=batch_size, shuffle=False, num_workers=4,
                                      collate_fn=collate_fn)

    num_classes = 21
    model = SSD(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = ssd_loss
    metric_fn = ssd_accuracy

    ssd_trainer = SSDTrainer(model=model,
                             loss=loss_fn,
                             optimizer=optimizer,
                             metric=metric_fn,
                             train_data_loader=train_data_loader,
                             valid_data_loader=valid_data_loader,
                             mac_gpu=True)

    train_loss, train_acc = ssd_trainer.train(epochs)
    val_loss, val_acc = ssd_trainer.validate()
