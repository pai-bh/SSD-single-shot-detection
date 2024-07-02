from base.base_data_loader import BaseDataLoader
from base.base_data_loader import BaseDataLoader


class CIFAR10DataLoader(BaseDataLoader):
    pass


class VOCDataLoader(BaseDataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=None):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                         collate_fn=collate_fn)
