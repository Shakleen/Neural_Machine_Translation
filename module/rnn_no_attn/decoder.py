import torch
from torch import nn

from ..base.decoder import Decoder as BaseDecoder
from .init_seq_2_seq import init_seq2seq


class Decoder(BaseDecoder):
    """The RNN decoder for seq2seq learning."""

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 num_hiddens: int,
                 num_layers: int,
                 dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim+num_hiddens,
                          num_hiddens, num_layers, dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(init_seq2seq)

    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def forward(self, X, state):
        # X shape: (batch_size, num_steps)
        # Embedding Shape: (num_steps, batch_size, embedding_dim)
        embs = self.embedding(X.to().type(torch.int64))
        enc_output, hidden_state = state
        # Context shape: (batch_size, num_hiddens)
        context = enc_output[-1]
        # Broadcast context: (num_steps, batch_size, num_hiddens)
        context = context.repeat(embs.shape[0], 1, 1)
        # Concat at the feature dimension
        embs_and_context = torch.cat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        outputs = self.dense(outputs).swapaxes(0, 1)
        # Outputs Shape: (batch_size, num_steps, vocab_size)
        # hidden_state Shape: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]
