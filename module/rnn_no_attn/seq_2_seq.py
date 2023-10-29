import torch
from torch.optim import Adam

from ..base.encoder_decoder import EncoderDecoder


class Seq2Seq(EncoderDecoder):
    """The RNN seq2seq model."""

    def __init__(self, encoder, decoder, tgt_pad, learning_rate):
        super().__init__(encoder, decoder)
        self.save_hyperparameters()

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    def loss(self, Y_hat, Y):
        loss = super(Seq2Seq, self).loss(Y_hat, Y, averaged=False)
        mask = (Y.reshape(-1) != self.tgt_pad).type(torch.float32)
        return (loss * mask).sum() / mask.sum()
