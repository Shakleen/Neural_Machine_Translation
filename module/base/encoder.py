from torch import nn


class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""

    def __init__(self):
        super().__init__()

    def forward(self, X, *args):
        raise NotImplementedError
