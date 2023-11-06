import math
import torch
from torch import nn

from .masked_softmax import masked_softmax


class DotProductAttention(nn.Module):
    """Scaled dot product attention"""

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """Performs scaled dot product attention.

        Args:
            queries (torch.Tensor): Shape (batch_size, no_of_queries, d)
            keys (torch.Tensor): Shape (batch_size, no_of_key_val_pairs, d)
            values (torch.Tensor): Shape (batch_size, no_of_key_val_pairs, val_dim)
            valid_lens (torch.Tensor, optional): Shape is either (batch_size,) or 
            (batch_size, no_of_queries). Defaults to None.

        Returns:
            torch.Tensor: Dot product attention scores.
        """
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
