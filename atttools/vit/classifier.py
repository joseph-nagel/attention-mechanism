'''Classifier head.'''

import torch.nn as nn


class ClassifierHead(nn.Module):
    '''
    ViT classification head.

    Parameters
    ----------
    embed_dim : int
        Number of embedding features.
    num_classes : int
        Number of target classes.

    '''

    def __init__(self, embed_dim, num_classes):
        super().__init__()

        self.ln = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        cls_token = x[:,0] # consider class token
        out = self.ln(cls_token)
        out = self.linear(out)
        return out

