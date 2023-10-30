import torch

from .hyperparameters import HyperParameters
from module.module import Module
from data_module.data_module import DataModule


class Trainer(HyperParameters):
    """The base class for training models with data."""

    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data: DataModule):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model: Module):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model: Module, data: DataModule):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = self.train_batch_idx = self.val_batch_idx = 0

        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        self.train_epoch()

        if self.val_dataloader is None:
            return

        self.val_epoch()

    def train_epoch(self):
        self.model.train()

        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()

            with torch.no_grad():
                loss.backward()

                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)

                self.optim.step()

            self.train_batch_idx += 1

    def val_epoch(self):
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))

            self.val_batch_idx += 1

    def prepare_batch(self, batch):
        return batch

    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm
