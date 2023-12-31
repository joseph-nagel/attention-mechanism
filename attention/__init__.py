'''
Attention mechanisms.

Summary
-------
Some components of the transformer in https://arxiv.org/abs/1706.03762 are implemented.
This involves the dot-product attention and the sinusoidal position encoding.

Modules
-------
encoding : Positional encoding.
dotprod : Dot-product attention.

'''

from .dotprod import (
    attend,
    self_attend,
    SelfAttention,
    SelfAttention2D
)

from .encoding import (
    make_frequencies,
    encode_position,
    make_encoding,
    SinusoidalEncoding
)

