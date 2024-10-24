'''ViT base module.'''

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
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
    lr : float
        Initial optimizer learning rate.
    warmup : int
        Warmup steps/epochs.

    '''

    def __init__(
        self,
        patchemb: nn.Module,
        encoder: nn.Module,
        head: nn.Module,
        lossfcn: nn.Module | Callable[[torch.Tensor], torch.Tensor],
        lr: float = 1e-04,
        warmup: int = 0
    ) -> None:

        super().__init__()

        # set models
        self.patchemb = patchemb
        self.encoder = encoder
        self.head = head
        self.lossfcn = lossfcn

        # set LR params
        self.lr = abs(lr)
        self.warmup = abs(int(warmup))

        # store hyperparams
        self.save_hyperparameters(
            ignore=['patchemb', 'encoder', 'head', 'lossfcn'],
            logger=True
        )

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        # run patch embedding
        x = self.patchemb(x)

        # run transformer encoder
        out = self.encoder(x, return_weights=return_weights)

        if return_weights:
            x, weights = out
        else:
            x = out

        # run prediction head
        x = self.head(x)

        if return_weights:
            return x, weights
        else:
            return x

    @staticmethod
    def _get_batch(
        batch: Sequence[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        '''Get batch features and labels.'''

        if isinstance(batch, Sequence):
            x_batch = batch[0]
            y_batch = batch[1]

        elif isinstance(batch, dict):
            x_batch = batch['features']
            y_batch = batch['labels']

        else:
            raise TypeError(f'Invalid batch type: {type(batch)}')

        return x_batch, y_batch

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''Compute the loss function.'''
        y_pred = self(x, return_weights=False)
        loss = self.lossfcn(y_pred, y)
        return loss

    def training_step(
        self,
        batch: Sequence[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        x_batch, y_batch = self._get_batch(batch)
        loss = self.loss(x_batch, y_batch)
        self.log('train_loss', loss.item()) # Lightning logs batch-wise scalars during training per default

        return loss

    def validation_step(
        self,
        batch: Sequence[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        x_batch, y_batch = self._get_batch(batch)
        loss = self.loss(x_batch, y_batch)
        self.log('val_loss', loss.item()) # Lightning automatically averages scalars over batches for validation

        return loss

    def test_step(
        self,
        batch: Sequence[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        x_batch, y_batch = self._get_batch(batch)
        loss = self.loss(x_batch, y_batch)
        self.log('test_loss', loss.item()) # Lightning automatically averages scalars over batches for testing

        return loss

    def configure_optimizers(self) -> tuple[list, list]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # create warmup schedule
        warmup = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda curr: ((curr + 1) / self.warmup) if curr < self.warmup else 1.0
        )

        # create reduce-on-plateau schedule
        # reduce = torch.optim.lr_scheduler.ReduceLROnPlateau( # SequentialLR cannot handle ReduceLROnPlateau
        #     optimizer=optimizer
        # )

        # create cosine annealing schedule
        annealing = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.trainer.max_epochs - self.warmup # consider remaining epochs
        )

        # create combined schedule
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, annealing],
            milestones=[self.warmup]
        )

        # lr_config = {
        #     'scheduler': lr_scheduler, # LR scheduler
        #     'interval': 'epoch', # time unit
        #     'frequency': 1 # update frequency
        # }

        return [optimizer], [lr_scheduler]

