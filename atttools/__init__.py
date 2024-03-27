'''
Attention tools.

Summary
-------
Some components of the transformer in https://arxiv.org/abs/1706.03762 are implemented.
This involves the dot-product attention and the sinusoidal position encoding.
An implementation of the vision transformer (ViT) that has been originally
proposed in the paper https://arxiv.org/abs/2010.11929 is provided.

Modules
-------
attention : Dot-product attention.
data : Datamodules.
encoding : Positional encoding.
vit : Vision transformer.

'''

from . import attention
from . import data
from . import encoding
from . import vit


from .attention import (
    attend,
    self_attend,
    SelfAttention,
    MultiheadSelfAttention,
    SelfAttention2D
)

from .data import FashionMNISTDataModule

from .encoding import (
    make_frequencies,
    encode_position,
    make_encoding,
    SinusoidalEncoding
)

from .vit import (
    ClassifierHead,
    EncoderBlock,
    Encoder,
    PatchEmbedding,
    BaseViT,
    ClassifierViT
)

