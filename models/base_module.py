import torch.nn.functional as F
import pytorch_lightning as pl
import torch


class Transformer_BaseModule(pl.LightningModule):
    """
    Base module for any pyTorch Lightning based algorithm in this library

    Parameters
    ----------
        in_features: int,
            The number of input features.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            A dictionary of metrics to be returned in the test phase. These metric are functions that accepts two tensors as
            inputs, i.e., predictions and targets, and return another tensor with the metric value. Defaults contains the accuracy

        top_module: nn.Module, defaults=None
            The optional nn.Module to be used as additional top layers.

        optimizer:  torch.optim.Optimizer
            The pyTorch Optimizer to use. Note that this must be only the class type and not an instance of the class!!

        **kwargs: dict
            A dictionary with the parameters of the optimizer.
    """

    def __init__(self, loss, optimizers, metrics, **kwargs):
        super(Transformer_BaseModule, self).__init__()
        self.kwargs = kwargs
        self.loss = loss
        self.optimizers_ = optimizers
        self.metrics = metrics
        pass

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        src,tgt,y = batch
        y_hat = self(src,tgt)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        #self.log('train_acc', self.accuracy(y_hat, y))
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        src,tgt,y = batch
        y_hat = self(src,tgt)
        val_loss = self.loss(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        src, tgt, y = batch
        y_hat = self(src,tgt)

        # compute test loss
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True)

        # compute the remaining test metrics
        for name, f in self.metrics.items():
            value = f(y_hat, y)
            self.log(str('test_' + name), value, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        opt = self.optimizers_(self.parameters(), **self.kwargs)
        return opt