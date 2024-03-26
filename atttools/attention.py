'''
Dot-product attention.

Summary
-------
The classical (scaled) dot-product attention variant is implemented.
A self-attention mechanism relates items at different positions of a sequence.
In comparison to recurrent architectures, it is parallelizable, does not severely suffer from
from vanishing/exploding gradients and allows for better capturing longe-range dependencies.
The latter is also an advantage over conv-layers with limited-size local receptive fields.

'''

import math

import torch
import torch.nn as nn


def attend(Q, K, V, scale=True):
    '''Compute (scaled) dot-product attention.'''

    # compute alignment scores
    algn_scores = torch.matmul(Q, K.transpose(-2, -1))

    if scale:
        d_k = K.shape[-1]
        algn_scores = algn_scores / math.sqrt(d_k)

    # compute attention weights
    attn_weights = nn.functional.softmax(algn_scores, dim=-1)

    # compute attention
    attn = torch.matmul(attn_weights, V)

    return attn


def self_attend(X, W_q, W_k, W_v, scale=True):
    '''Compute (scaled) dot-product self-attention.'''

    # compute queries, keys and values
    Q = torch.matmul(X, W_q)
    K = torch.matmul(X, W_k)
    V = torch.matmul(X, W_v)

    # compute attention
    attn = attend(Q, K, V, scale=scale)

    return attn


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
    d_k : int
        Number of queries and keys.
    d_v : int
        Number of values.
    scale : bool
        Determines whether scores are scaled.

    '''

    def __init__(self,
                 d_x,
                 d_k=None,
                 d_v=None,
                 scale=True):

        super().__init__()

        if d_k is None:
            d_k = d_x

        if d_v is None:
            d_v = d_x

        self.q = nn.Linear(d_x, d_k, bias=False) # query
        self.k = nn.Linear(d_x, d_k, bias=False) # key
        self.v = nn.Linear(d_x, d_v, bias=False) # value

        self.scale = scale

    def forward(self, x):

        # ensure (batch, sequence, features)-shaped input
        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim != 3:
            raise ValueError('Invalid number of tensor dimensions: {}'.format(x.ndim))

        # compute queries, keys and values
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        # compute attention
        attn = nn.functional.scaled_dot_product_attention(
            Q, K, V,
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
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    num_heads : int
        Number of attention heads.
    scale : bool
        Determines whether scores are scaled.

    '''

    def __init__(self,
                 in_features,
                 out_features,
                 num_heads,
                 scale=True):

        super().__init__()

        # create attention heads
        heads = [
            SelfAttention(
                d_x=in_features,
                d_k=in_features,
                d_v=in_features,
                scale=scale
            ) for _ in range(num_heads)
        ]

        self.heads = nn.ModuleList(heads)

        # create linear layer
        self.linear = nn.Linear(num_heads * in_features, out_features)

    def forward(self, x):

        # ensure (batch, sequence, features)-shaped input
        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim != 3:
            raise ValueError('Invalid number of tensor dimensions: {}'.format(x.ndim))

        # run attention heads
        x = torch.cat([h(x) for h in self.heads], dim=-1)

        # transform linearly
        x = self.linear(x)

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
    in_channels : int
        Number of input and output channels.
    out_channels : int
        Number of queries and keys.
    scale : bool
        Determines whether scores are scaled.

    '''

    def __init__(self,
                 in_channels,
                 out_channels=None,
                 scale=False):

        super().__init__()

        if out_channels is None:
            out_channels = in_channels // 8 # set to the default value in the paper

        self.f = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False) # query
        self.g = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False) # key
        self.h = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False) # value

        self.gamma = nn.Parameter(torch.tensor(0.0))

        if scale:
            d_k_sqrt = torch.tensor(out_channels).sqrt()
            self.register_buffer('scale', d_k_sqrt)
        else:
            self.scale = None

    def forward(self, x):
        b, c, h, w = x.shape

        # flatten tensor (last axis contains the sequence)
        x_flattened = x.view(b, c, h*w) # (b, c, h*w)

        # compute query, key and value
        q = self.f(x_flattened) # (b, c', h*w)
        k = self.g(x_flattened) # (b, c', h*w)
        v = self.h(x_flattened) # (b, c, h*w)

        # compute attention
        algn_scores = torch.bmm(q.transpose(1, 2), k) # (b, h*w, h*w)

        if self.scale is not None:
            algn_scores = algn_scores / self.scale

        attn_weights = torch.softmax(algn_scores, dim=1) # (b, h*w, h*w)

        attention = torch.bmm(v, attn_weights) # (b, c, h*w)

        # add skip connection
        out = self.gamma * attention + x_flattened # (b, c, h*w)

        # reshape
        out = out.view(b, c, h, w) # (b, c, h, w)

        return out

