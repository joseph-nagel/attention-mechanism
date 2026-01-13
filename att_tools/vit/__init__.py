'''
Vision transformer.

Summary
-------
The vision transformer (ViT) is implemented from scratch.
For the moment, the focus is on classification problems.

Modules
-------
base : ViT base module.
classifier: ViT classifier.
encoder : ViT encoder.
patches : Patch embedding.

'''

from . import (
    base,
    classifier,
    encoder,
    patches
)
from .base import BaseViT
from .classifier import ClassifierHead, ClassifierViT
from .encoder import EncoderBlock, Encoder
from .patches import PatchEmbedding
