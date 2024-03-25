'''
Attention mechanisms.

Summary
-------
Some components of the transformer in https://arxiv.org/abs/1706.03762 are implemented.
This involves the dot-product attention and the sinusoidal position encoding.
Moreover, am implementation of the vision transformer (ViT) originally
proposed in the paper https://arxiv.org/abs/2010.11929 is provided.

Modules
-------
dotprod : Dot-product attention.
encoding : Positional encoding.
vit : Vision transformer.

'''

from . import dotprod
from . import encoding
from . import vit


from .dotprod import (
    attend,
    self_attend,
    SelfAttention,
    MultiheadSelfAttention,
    SelfAttention2D
)

from .encoding import (
    make_frequencies,
    encode_position,
    make_encoding,
    SinusoidalEncoding
)

from .vit import PatchEmbedding

