'''ViT classifier.'''

import torch.nn as nn

from .base import BaseViT
from .encoder import Encoder
from .patches import PatchEmbedding


class ClassifierHead(nn.Module):
    '''
    ViT classification head.

    Parameters
    ----------
    embed_dim : int
        Number of embedding features.
    num_classes : int
        Number of target classes.

    '''

    def __init__(self, embed_dim, num_classes):
        super().__init__()

        self.ln = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        cls_token = x[:,0] # consider class token
        out = self.ln(cls_token)
        out = self.linear(out)
        return out


class ClassifierViT(BaseViT):
    '''
    ViT classifier module.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    embed_dim : int
        Number of embedding features.
    num_classes : int
        Number of target classes.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of encoder blocks.
    num_patches : int, optional
        Prefixed number of patches, required for pos. embedding.
    patch_size : int
        Size of the patches.
    mlp_dim : int, optional
        MLP hidden dimensionality.
    mlp_dropout : float, optional
        MLP dropout rate.
    lr : float, optional
        Initial optimizer learning rate.

    '''

    def __init__(self,
                 in_channels,
                 embed_dim,
                 num_classes,
                 num_heads,
                 num_blocks,
                 num_patches,
                 patch_size,
                 mlp_dim=None,
                 mlp_dropout=0.0,
                 lr=1e-04):

        # create patch embedding
        patchemb = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            use_cls_token=True,
            use_pos_embedding=True,
            num_patches=num_patches
        )

        # create encoder
        encoder = Encoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            mlp_dim=mlp_dim,
            mlp_dropout=mlp_dropout,
            use_custom_mha=False
        )

        # create classifier head
        classifier = ClassifierHead(
            embed_dim=embed_dim,
            num_classes=num_classes
        )

        # create loss function
        lossfcn = nn.CrossEntropyLoss(reduction='mean')

        # initialize embedding class
        super().__init__(
            patchemb=patchemb,
            encoder=encoder,
            head=classifier,
            lossfcn=lossfcn,
            lr=lr
        )

        # store hyperparams
        self.save_hyperparameters(logger=True)

