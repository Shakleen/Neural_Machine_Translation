from .decoder import Decoder


class AttentionDecoder(Decoder):
    """The base attention-based decoder interface."""

    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError
