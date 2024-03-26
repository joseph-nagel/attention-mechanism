'''ViT encoder.'''

import torch.nn as nn

from ..attention import MultiheadSelfAttention


class EncoderBlock(nn.Module):
    '''ViT encoder block.'''

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

    def forward(self, x):

        # run attention block
        x = self.ln1(x)

        if self.use_custom_mha:
            x = x + self.att(x)
        else:
            x = x + self.att(x, x, x, need_weights=False)[0]

        # run MLP block
        x = self.ln2(x)
        x = x + self.mlp(x)

        return x


class Encoder(nn.Sequential):
    '''ViT encoder.'''

    def __init__(self,
                 embed_dim,
                 num_heads,
                 num_blocks,
                 mlp_dim=None,
                 mlp_dropout=0.0,
                 use_custom_mha=False):

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

        # initialize module
        super().__init__(*blocks)

