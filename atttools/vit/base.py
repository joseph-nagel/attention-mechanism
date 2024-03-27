'''ViT base module.'''

import torch
from lightning.pytorch import LightningModule


class BaseViT(LightningModule):
    '''
    Base ViT module.

    Parameters
    ----------
    patchemb : PyTorch module
        Patch embedding.
    encoder : PyTorch module
        ViT encoder.
    head : PyTorch module
        Prediction head.
    lossfcn : PyTorch module or callable
        Loss function.
    lr : float, optional
        Initial optimizer learning rate.

    '''

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
        self.log('train_loss', loss.item()) # Lightning logs batch-wise scalars during training per default
        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = self._get_batch(batch)
        loss = self.loss(x_batch, y_batch)
        self.log('val_loss', loss.item()) # Lightning automatically averages scalars over batches for validation
        return loss

    def test_step(self, batch, batch_idx):
        x_batch, y_batch = self._get_batch(batch)
        loss = self.loss(x_batch, y_batch)
        self.log('test_loss', loss.item()) # Lightning automatically averages scalars over batches for testing
        return loss

    # TODO: enable LR scheduling
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

