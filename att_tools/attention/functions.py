'''Attention functions.'''

import math

import torch
import torch.nn as nn


def attend(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: bool = True
) -> torch.Tensor:
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


def self_attend(
    X: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    scale: bool = True
) -> torch.Tensor:
    '''Compute (scaled) dot-product self-attention.'''

    # compute queries, keys and values
    Q = torch.matmul(X, W_q)
    K = torch.matmul(X, W_k)
    V = torch.matmul(X, W_v)

    # compute attention
    attn = attend(Q, K, V, scale=scale)

    return attn

