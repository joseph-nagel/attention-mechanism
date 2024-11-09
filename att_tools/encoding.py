'''
Positional encoding.

Summary
-------
This module implements the sinusoidal embedding from the original transformer architecture.
It can be used in order to encode spatial positions or times and ingest them in further layers.
For a (batch_size, 1)-shaped input, the (batch_size, embed_dim)-sized embedding is computed.

'''

import torch
import torch.nn as nn


def make_frequencies(embed_dim: int) -> torch.Tensor:
    '''Create angular frequencies.'''

    # create frequencies
    i = torch.arange(embed_dim // 2)
    omega = 1 / (10000 ** (2 * i / embed_dim))

    # reshape into (1, embed_dim // 2)
    omega = omega.view(1, -1)

    return omega


def encode_position(
    t: torch.Tensor,
    embed_dim: int,
    omega: torch.Tensor | None = None
) -> torch.Tensor:
    '''Compute sinusoidal encoding.'''

    # create frequencies if not passed
    if omega is None:
        omega = make_frequencies(embed_dim)

    # ensure (batch_size>=1, 1)-shaped tensor
    if t.numel() == 1:
        t = t.view(1, 1)
    elif t.ndim != 2 or t.shape[1] != 1:
        raise ValueError('Invalid shape encountered: {}'.format(t.shape))

    # compute and assemble embeddings
    device = t.device
    batch_size = t.shape[0]

    emb = torch.zeros(batch_size, embed_dim, device=device)

    emb[:,0::2] = torch.sin(omega * t)
    emb[:,1::2] = torch.cos(omega * t)

    return emb


def make_encoding(max_length: int, embed_dim: int) -> nn.Embedding:
    '''
    Create sinusoidal encoding lookup table.

    Parameters
    ----------
    max_length : int
        Maximum sequence length.
    embed_dim : int
        Embedding dimensionality.

    '''

    # create embedding matrix
    t = torch.arange(max_length).view(-1, 1)

    embed_mat = encode_position(
        t=t,
        embed_dim=embed_dim,
    )

    # create lookup table
    lookup_table = nn.Embedding.from_pretrained(
        embeddings=embed_mat,
        freeze=True
    )

    return lookup_table


class SinusoidalEncoding(nn.Module):
    '''
    Sinusoidal position encoding.

    Parameters
    ----------
    embed_dim : int
        Embedding dimensionality.

    '''

    def __init__(self, embed_dim: int) -> None:

        super().__init__()

        # set embedding dimension
        embed_dim = abs(embed_dim)

        if embed_dim < 2:
            raise ValueError('At least two embedding dimensions required')
        elif embed_dim % 2 != 0:
            raise ValueError('Dimensionality has to be an even number')

        self.embed_dim = embed_dim

        # create angular frequencies
        omega = make_frequencies(embed_dim)

        self.register_buffer('omega', omega)

    def forward(self, t: torch.Tensor) -> torch.Tensor:

        emb = encode_position(
            t=t,
            embed_dim=self.embed_dim,
            omega=self.omega
        )

        return emb

