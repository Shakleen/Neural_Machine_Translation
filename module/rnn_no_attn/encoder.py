import torch
from torch import nn

from ..base.encoder import Encoder as BaseEncoder
from .init_seq_2_seq import init_seq2seq


class Encoder(BaseEncoder):
    """The RNN encoder for seq2seq learning."""

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 num_hiddens: int,
                 num_layers: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, num_hiddens, num_layers, dropout)
        self.apply(init_seq2seq)

    def forward(self, X: torch.Tensor, *args):
        # X shape: (batch_size, num_steps)
        # Embedding shape: (num_steps, batch_size, embedding_dim)
        emb = self.embedding(X.to().type(torch.int64))
        # Output shape: (num_steps, batch_size, num_hiddens)
        # States shape: (num_layers, batch_sie, num_hiddens)
        output, states = self.rnn(emb)
        return output, states
