'''Vision transformer.'''

import torch.nn as nn

from .encoder import Encoder
from .patches import PatchEmbedding


class ViT(nn.Module):
    '''ViT module.'''

    def __init__(self,
                 in_channels,
                 embed_dim,
                 patch_size,
                 num_patches,
                 num_heads,
                 num_blocks,
                 mlp_dim=None,
                 mlp_dropout=0.0,
                 use_custom_mha=False,
                 use_cls_token=True):

        super().__init__()

        # create patch embedding
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            use_cls_token=use_cls_token,
            use_pos_embedding=True,
            num_patches=num_patches
        )

        # create encoder
        self.encoder = Encoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            mlp_dim=mlp_dim,
            mlp_dropout=mlp_dropout,
            use_custom_mha=use_custom_mha
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        return x

