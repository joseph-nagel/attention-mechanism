'''Attention layers.'''

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    '''
    Dot-product self-attention for sequential data.

    Summary
    -------
    The layer realizes the classical dot-product self-attention.
    It operates on (batch, sequence, features)-shaped input tensors.

    Parameters
    ----------
    d_x : int
        Number of input features.
    d_k : int or None
        Number of queries and keys.
    d_v : int or None
        Number of values.
    scale : bool
        Determines whether scores are scaled.

    '''

    def __init__(
        self,
        d_x: int,
        d_k: int | None = None,
        d_v: int | None = None,
        scale: bool = True
    ) -> None:

        super().__init__()

        if d_k is None:
            d_k = d_x

        if d_v is None:
            d_v = d_x

        self.q = nn.Linear(d_x, d_k, bias=False)  # queries
        self.k = nn.Linear(d_x, d_k, bias=False)  # keys
        self.v = nn.Linear(d_x, d_v, bias=False)  # values

        self.scale = scale

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False
    ) -> torch.Tensor:

        # ensure (batch, sequence, features)-shaped input
        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim != 3:
            raise ValueError(f'Invalid number of tensor dimensions: {x.ndim}')

        # compute queries, keys and values
        q = self.q(x)  # (batch, sequence, d_k)
        k = self.k(x)  # (batch, sequence, d_k)
        v = self.v(x)  # (batch, sequence, d_v)

        # compute attention
        attn = nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            scale=None if self.scale else 1.0
        )

        return attn


class MultiheadSelfAttention(nn.Module):
    '''
    Multihead self-attention.

    Summary
    -------
    A multihead version of self-attention is implemented.
    It simply uses the self-attention layer from above
    in order to represent multiple attention heads.
    The outputs are concatenated and linearly transformed.

    Parameters
    ----------
    embed_dim : int
        Number of input and output features.
    num_heads : int
        Number of attention heads.
    scale : bool
        Determines whether scores are scaled.

    '''

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        scale: bool = True
    ) -> None:

        super().__init__()

        # consider dimensionality
        if embed_dim % num_heads == 0:
            head_dim = embed_dim // num_heads
        else:
            raise ValueError('Embedding dim. must be divisible by head number')

        # create attention heads
        heads = [
            SelfAttention(
                d_x=embed_dim,  # input dim. d_x
                d_k=head_dim,  # intermediate dims. with d_q = d_k
                d_v=head_dim,  # output dim. d_v
                scale=scale
            ) for _ in range(num_heads)
        ]

        self.heads = nn.ModuleList(heads)

        # create linear layer
        # self.linear = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False
    ) -> torch.Tensor:

        # ensure (batch, sequence, features)-shaped input
        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim != 3:
            raise ValueError(f'Invalid number of tensor dimensions: {x.ndim}')

        # run attention heads
        x = torch.cat(
            [h(x, attn_mask=attn_mask, is_causal=is_causal) for h in self.heads],
            dim=-1
        )

        # run linear layer
        # x = self.linear(x)

        return x


class SelfAttention2D(nn.Module):
    '''
    Self-attention with skip connections for 2D data.

    Summary
    -------
    This module establishes the self-attention from https://arxiv.org/abs/1805.08318.
    It employs a residual skip connection adding the inputs after the attention.
    The input shape for this layer is (batch, channels, height, width).

    Parameters
    ----------
    num_channels : int
        Number of input and output channels.
    num_queries_and_keys : int or None
        Number of queries and keys.
    scale : bool
        Determines whether scores are scaled.

    '''

    def __init__(
        self,
        num_channels: int,
        num_queries_and_keys: int | None = None,
        scale: bool = False
    ) -> None:

        super().__init__()

        if num_queries_and_keys is None:
            num_queries_and_keys = num_channels // 8  # set to the default value in the paper

        # create layer predicting queries
        self.q = nn.Conv1d(
            num_channels,
            num_queries_and_keys,
            kernel_size=1,
            bias=False
        )

        # create layer predicting keys
        self.k = nn.Conv1d(
            num_channels,
            num_queries_and_keys,
            kernel_size=1,
            bias=False
        )

        # create layer predicting values
        self.v = nn.Conv1d(
            num_channels,
            num_channels,
            kernel_size=1,
            bias=False
        )

        # initialize attention strength param
        self.gamma = nn.Parameter(torch.tensor(0.0))

        # initialize scaling factor
        if scale:
            d_k_sqrt = torch.tensor(num_queries_and_keys).sqrt()
            self.register_buffer('scale', d_k_sqrt)
        else:
            self.scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # get input dimensions
        b, c, h, w = x.shape

        # flatten tensor (last axis contains the sequence)
        x_flattened = x.view(b, c, h*w)  # (b, c, h*w)

        # compute query, key and value
        q = self.q(x_flattened)  # (b, c', h*w)
        k = self.k(x_flattened)  # (b, c', h*w)
        v = self.v(x_flattened)  # (b, c, h*w)

        # compute attention
        algn_scores = torch.bmm(q.transpose(1, 2), k)  # (b, h*w, h*w)

        if self.scale is not None:
            algn_scores = algn_scores / self.scale

        attn_weights = torch.softmax(algn_scores, dim=1)  # (b, h*w, h*w)

        attention = torch.bmm(v, attn_weights)  # (b, c, h*w)

        # add skip connection
        out = self.gamma * attention + x_flattened  # (b, c, h*w)

        # reshape
        out = out.view(b, c, h, w)  # (b, c, h, w)

        return out

