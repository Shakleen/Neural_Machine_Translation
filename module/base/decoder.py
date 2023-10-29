from torch import nn


class Decoder(nn.Module):  # @save
    """The base decoder interface for the encoder-decoder architecture."""

    def __init__(self):
        super().__init__()

    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
