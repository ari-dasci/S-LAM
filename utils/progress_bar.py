from tqdm import tqdm
import pytorch_lightning as pl

class SimpleProgressBar(pl.callbacks.ProgressBar):
    def __init__(self, model_name=None):
        super().__init__()
        self.bar = None
        self.enabled = True
        self.model_name = model_name

    def on_train_epoch_start(self, trainer, pl_module):
        if self.enabled:
            name = self.model_name if self.model_name!=None else ""
            self.bar = tqdm(total=self.total_train_batches,
                            desc=f"Epoch {trainer.current_epoch+1}, " + name,
                            position=0,
                            leave=True)
            self.running_loss = 0.0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.bar:
            self.running_loss += outputs['loss'].item()
            self.bar.update(1)
            loss = self.running_loss / self.total_train_batches
            self.bar.set_postfix(loss=f'{loss:.10f}')
            # self.bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.bar:
            val_loss = trainer.logged_metrics['val_loss'].item()
            loss = self.running_loss / self.total_train_batches
            self.bar.set_postfix(loss=f'{loss:.10f}', val_loss=f'{val_loss:.10f}')
            self.bar.close()
            self.bar = None

    def disable(self):
        self.bar = None
        self.enabled = False