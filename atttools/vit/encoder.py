'''ViT encoder.'''

import torch
import torch.nn as nn

from ..attention import MultiheadSelfAttention


class EncoderBlock(nn.Module):
    '''
    ViT encoder block.

    Parameters
    ----------
    embed_dim : int
        Number of embedding features.
    num_heads : int
        Number of attention heads.
    mlp_dim : int, optional
        MLP hidden dimensionality.
    mlp_dropout : float, optional
        MLP dropout rate.
    use_custom_mha : bool, optional
        Determines whether a custom or a native Pytorch
        implementation of multihead attention is used.

    '''

    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_dim=None,
                 mlp_dropout=0.0,
                 use_custom_mha=False):

        super().__init__()

        self.use_custom_mha = use_custom_mha

        if mlp_dim is None:
            mlp_dim = embed_dim

        # create attention block
        self.ln1 = nn.LayerNorm(embed_dim)

        if use_custom_mha:
            # use custom implementation
            self.att = MultiheadSelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                scale=True
            )
        else:
            # use PyTorch implementation
            self.att = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True
            )

        # create MLP block
        self.ln2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(mlp_dropout)
        )

    def forward(self, x, return_weights=False):

        # run attention block
        y = self.ln1(x)

        if self.use_custom_mha:
            if return_weights:
                raise NotImplementedError('Returning the weights from custom attention is not implemented')

            vals = self.att(y)

        else:
            out = self.att(y, y, y, need_weights=return_weights)
            vals = out[0]

            if return_weights:
                weights = out[1]

        y = x + vals

        # run MLP block
        z = self.ln2(y)
        z = y + self.mlp(z)

        if return_weights:
            return z, weights
        else:
            return z


class Encoder(nn.Module):
    '''
    ViT encoder.

    Parameters
    ----------
    embed_dim : int
        Number of embedding features.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of encoder blocks.
    mlp_dim : int, optional
        MLP hidden dimensionality.
    mlp_dropout : float, optional
        MLP dropout rate.
    use_custom_mha : bool, optional
        Determines whether a custom or a native Pytorch
        implementation of multihead attention is used.

    '''

    def __init__(self,
                 embed_dim,
                 num_heads,
                 num_blocks,
                 mlp_dim=None,
                 mlp_dropout=0.0,
                 use_custom_mha=False):

        super().__init__()

        # create encoder blocks
        blocks = [
            EncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                mlp_dropout=mlp_dropout,
                use_custom_mha=use_custom_mha

            ) for _ in range(num_blocks)
        ]

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, return_weights=False):

        if return_weights:
            weights = []

        # run encoder blocks
        for b in self.blocks:
            out = b(x, return_weights=return_weights)

            if return_weights:
                x, w = out
                weights.append(w)
            else:
                x = out

        if return_weights:
            weights = torch.cat([w.unsqueeze(1) for w in weights], dim=1)
            return x, weights
        else:
            return x

