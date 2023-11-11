'''Dot-product attention.'''

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
    '''(Scaled) dot-product self-attention layer.'''

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

