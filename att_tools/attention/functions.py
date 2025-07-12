'''Attention functions.'''

import math

import torch
import torch.nn as nn


def attend(
    Q: torch.Tensor,  # (m, d_k)
    K: torch.Tensor,  # (n, d_k)
    V: torch.Tensor,  # (n, d_v)
    scale: bool = True
) -> torch.Tensor:
    '''Compute (scaled) dot-product attention.'''

    # compute alignment scores
    algn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (m, n)

    if scale:
        d_k = K.shape[-1]
        algn_scores = algn_scores / math.sqrt(d_k)  # (m, n)

    # compute attention weights
    attn_weights = nn.functional.softmax(algn_scores, dim=-1)  # (m, n)

    # compute attention
    attn = torch.matmul(attn_weights, V)  # (m, d_v)

    return attn


def self_attend(
    X: torch.Tensor,  # (m, d_x)
    W_q: torch.Tensor,  # (d_x, d_k)
    W_k: torch.Tensor,  # (d_x, d_k)
    W_v: torch.Tensor,  # (d_x, d_v)
    scale: bool = True
) -> torch.Tensor:
    '''Compute (scaled) dot-product self-attention.'''

    # compute queries, keys and values
    Q = torch.matmul(X, W_q)  # (m, d_k)
    K = torch.matmul(X, W_k)  # (m, d_k)
    V = torch.matmul(X, W_v)  # (m, d_v)

    # compute attention
    attn = attend(Q, K, V, scale=scale)  # (m, d_v)

    return attn

