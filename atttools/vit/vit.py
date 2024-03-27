'''Vision transformer.'''

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from .classifier import ClassifierHead
from .encoder import Encoder
from .patches import PatchEmbedding


class BaseViT(LightningModule):
    '''Base ViT module.'''

    def __init__(self,
                 patchemb,
                 encoder,
                 head,
                 lossfcn,
                 lr=1e-04):

        super().__init__()

        # set models
        self.patchemb = patchemb
        self.encoder = encoder
        self.head = head
        self.lossfcn = lossfcn

        # set initial learning rate
        self.lr = abs(lr)

        # store hyperparams
        self.save_hyperparameters(
            ignore=['patchemb', 'encoder', 'head', 'lossfcn'],
            logger=True
        )

    def forward(self, x):
        x = self.patchemb(x)
        x = self.encoder(x)
        x = self.head(x)
        return x

    @staticmethod
    def _get_batch(batch):
        '''Get batch features and labels.'''

        if isinstance(batch, (tuple, list)):
            x_batch = batch[0]
            y_batch = batch[1]

        elif isinstance(batch, dict):
            x_batch = batch['features']
            y_batch = batch['labels']

        else:
            raise TypeError(f'Invalid batch type: {type(batch)}')

        return x_batch, y_batch

    def loss(self, x, y):
        '''Compute the loss function.'''
        y_pred = self(x)
        loss = self.lossfcn(y_pred, y)
        return loss

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = self._get_batch(batch)
        loss = self.loss(x_batch, y_batch)
        self.log('train_loss', loss.item()) # Lightning logs batch-wise metrics during training per default
        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = self._get_batch(batch)
        loss = self.loss(x_batch, y_batch)
        self.log('val_loss', loss.item()) # Lightning automatically averages metrics over batches for validation
        return loss

    def test_step(self, batch, batch_idx):
        x_batch, y_batch = self._get_batch(batch)
        loss = self.loss(x_batch, y_batch)
        self.log('test_loss', loss.item()) # Lightning automatically averages metrics over batches for testing
        return loss

    # TODO: enable LR scheduling
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class ClassifierViT(BaseViT):
    '''Classifier ViT module.'''

    def __init__(self,
                 in_channels,
                 embed_dim,
                 num_classes,
                 num_heads,
                 num_blocks,
                 num_patches,
                 patch_size,
                 mlp_dim=None,
                 mlp_dropout=0.0,
                 lr=1e-04):

        # create patch embedding
        patchemb = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            use_cls_token=True,
            use_pos_embedding=True,
            num_patches=num_patches
        )

        # create encoder
        encoder = Encoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            mlp_dim=mlp_dim,
            mlp_dropout=mlp_dropout,
            use_custom_mha=False
        )

        # create classifier head
        classifier = ClassifierHead(
            embed_dim=embed_dim,
            num_classes=num_classes
        )

        # create loss function
        lossfcn = nn.CrossEntropyLoss(reduction='mean')

        # initialize embedding class
        super().__init__(
            patchemb=patchemb,
            encoder=encoder,
            head=classifier,
            lossfcn=lossfcn,
            lr=lr
        )

        # store hyperparams
        self.save_hyperparameters(logger=True)

