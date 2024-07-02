# train/trainer.py
import torch
from base.base_trainer import BaseTrainer
from tqdm import tqdm

class SSDTrainer(BaseTrainer):
    """
    SSD 모델을 위한 트레이너 클래스
    """

    def __init__(self, model, loss, optimizer, metric, train_data_loader, valid_data_loader, mac_gpu=True, *args, **kwargs):
        """
        SSDTrainer 초기화

        Args:
            model (torch.nn.Module): 학습할 모델
            loss (function): 손실 함수
            optimizer (torch.optim.Optimizer): 옵티마이저
            metric (function): 메트릭 함수
            train_data_loader (torch.utils.data.DataLoader): 학습 데이터 로더
            valid_data_loader (torch.utils.data.DataLoader): 검증 데이터 로더
            mac_gpu (bool): GPU 사용 여부
        """
        super(SSDTrainer, self).__init__(model, loss, optimizer, metric, *args, **kwargs)
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.mac_gpu = mac_gpu

        if self.mac_gpu:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.model = self.model.to(self.device)

    def _train_epoch(self, epoch):
        """
        한 에폭 동안의 학습 루프

        Args:
            epoch (int): 현재 에폭

        Returns:
            tuple: 에폭 손실 및 정확도
        """
        batch_loss = 0
        batch_total = 0
        batch_correct = 0
        self.model.train()

        for inputs, targets in tqdm(self.train_data_loader):
            if self.mac_gpu:
                inputs = inputs.to(self.device)
                targets = {'boxes': [t.to(self.device) for t in targets['boxes']],
                           'labels': [t.to(self.device) for t in targets['labels']]}

            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.loss(outputs, targets, self.model.num_classes)
            loss.backward()

            self.optimizer.step()

            batch_loss += loss.item() * inputs.size(0)
            total, correct = self.metric(outputs, targets)
            batch_total += total
            batch_correct += correct

        epoch_loss = batch_loss / len(self.train_data_loader.dataset)
        epoch_accuracy = 100 * batch_correct / batch_total

        return epoch_loss, epoch_accuracy

    def train(self, epochs):
        """
        주어진 에폭 수만큼 모델 학습 수행

        Args:
            epochs (int): 학습할 에폭 수

        Returns:
            tuple: 최종 에폭 손실 및 정확도
        """
        print(f'{epochs} 번의 학습 시작')
        for epoch in range(epochs):
            epoch_loss, epoch_accuracy = self._train_epoch(epoch)
            print(f'Epoch: {epoch} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%')
        print(f'{epochs} 번의 학습 완료')
        return epoch_loss, epoch_accuracy

    def validate(self):
        """
        모델 검증 수행

        Returns:
            tuple: 검증 손실 및 정확도
        """
        val_loss = 0
        total = 0
        correct = 0

        self.model.eval()
        with torch.no_grad():
            for inputs, targets in self.valid_data_loader:
                if self.mac_gpu:
                    inputs = inputs.to(self.device)
                    targets = {'boxes': [t.to(self.device) for t in targets['boxes']],
                               'labels': [t.to(self.device) for t in targets['labels']]}

                outputs = self.model(inputs)

                loss = self.loss(outputs, targets, self.model.num_classes)

                val_loss += loss.item() * inputs.size(0)
                val_total, val_correct = self.metric(outputs, targets)
                total += val_total
                correct += val_correct

        val_acc = 100 * correct / total
        return val_loss, val_acc
