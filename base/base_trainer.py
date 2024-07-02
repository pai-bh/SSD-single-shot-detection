import torch
from abc import abstractmethod


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, *args, **kwargs):
        self.model = model

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        n_epochs 만큼 학습이 돌아갈 때 학습 full logic
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, epochs):
        raise NotImplementedError
