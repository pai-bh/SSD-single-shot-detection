import torch
from abc import abstractmethod
from torch.utils.data import Dataset


class BaseDataSetter(Dataset):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError
