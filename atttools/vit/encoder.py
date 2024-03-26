'''ViT encoder.'''

import torch.nn as nn

from ..attention import MultiheadSelfAttention


class EncoderBlock(nn.Module):
    '''ViT encoder block.'''

    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_dim=None,
                 mlp_dropout=0.0):

        super().__init__()

        # create attention block
        self.ln1 = nn.LayerNorm(embed_dim)

        self.att = MultiheadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            scale=True
        )

        # create MLP block
        self.ln2 = nn.LayerNorm(embed_dim)

        if mlp_dim is None:
            mlp_dim = embed_dim

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
        x = x + self.att(x)

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
                 mlp_dropout=0.0):

        # create encoder blocks
        blocks = [
            EncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                mlp_dropout=mlp_dropout

            ) for _ in range(num_blocks)
        ]

        # initialize module
        super().__init__(*blocks)

