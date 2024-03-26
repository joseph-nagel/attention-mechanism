'''Patch embedding.'''

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    '''
    Patch embedding module.

    Summary
    -------
    Input images are organized into patches, flattened,
    and linearly transformed into an embedding space.
    This is realized through an appropriate conv. layer.
    An optional learnable position embedding may be used.
    A learnable class token can be included similarly.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    patch_size : int
        Size of the patches.
    use_cls_token : bool, optional
        Determines whether a BERT-like class token is utilized.
    use_pos_embedding : bool, optional
        Determines whether an additional pos. embedding is learned.
    num_patches : int, optional
        Prefixed number of patches, required for pos. embedding.

    '''

    def __init__(self,
                 in_channels,
                 out_features,
                 patch_size,
                 use_cls_token=False,
                 use_pos_embedding=False,
                 num_patches=None):

        super().__init__()

        self.in_channels = in_channels
        self.out_features = out_features
        self.patch_size = patch_size

        # create patch embedding as conv layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_features,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )

        # create learnable class token embedding
        if use_cls_token:
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, out_features), # (1, 1, c)
                requires_grad=True
            )
        else:
            self.cls_token = None

        # create learnable positional embedding
        if use_pos_embedding:
            if num_patches is not None:
                self.pos_embedding = nn.Parameter(
                    torch.randn(1, num_patches + 1 if use_cls_token else num_patches, out_features), # (b, p(+1), c)
                    requires_grad=True
                )
            else:
                raise TypeError('Number of patches is missing')
        else:
            self.pos_embedding = None

    def forward(self, x):

        # embed patches
        x = self.conv(x) # (b, c, h', w')
        x = x.flatten(start_dim=2) # (b, c, p), with p = h'*w'
        x = x.transpose(1, 2) # (b, p, c)

        # concatenate class token
        if self.cls_token is not None:
            cls_token = self.cls_token # (1, 1, c)
            cls_token = cls_token.expand(x.shape[0], -1, -1) # (b, 1, c)
            x = torch.cat((cls_token, x), dim=1) # (b, p(+1), c)

        # add position embedding
        if self.pos_embedding is not None:
            pos_embedding = self.pos_embedding # (1, p(+1), c)
            x = x + pos_embedding # (b, p(+1), c)

        return x

