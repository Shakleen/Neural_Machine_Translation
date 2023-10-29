from torch.utils.data import DataLoader, TensorDataset

from utils.hyperparameters import HyperParameters


class DataModule(HyperParameters):  # @save
    """The base class of data."""

    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train: bool):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train: bool,
                         indices=slice(0, None)) -> DataLoader:
        tensors = tuple([a[indices] for a in tensors])
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, self.batch_size, shuffle=train)
