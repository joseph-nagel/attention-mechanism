'''
Dot-product attention.

Summary
-------
The classical (scaled) dot-product attention variant is implemented.
A self-attention mechanism relates items at different positions of a sequence.
In comparison to recurrent architectures, it is parallelizable, does not severely suffer from
from vanishing/exploding gradients and allows for better capturing longe-range dependencies.
The latter is also an advantage over conv-layers with limited-size local receptive fields.

Modules
-------
functions : Attention functions.
layers : Attention layers.

'''

from . import functions, layers

from .functions import attend, self_attend

from .layers import (
    SelfAttention,
    MultiheadSelfAttention,
    SelfAttention2D
)

