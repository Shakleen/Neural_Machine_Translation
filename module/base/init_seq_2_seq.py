from torch import nn


def init_seq2seq(module):
    """Used to initialize weights using Xavier initialization.

    Args:
        module (nn.Module): The module whose weights should be 
        initialized.
    """
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)

    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])
