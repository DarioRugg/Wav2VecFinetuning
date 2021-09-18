from pytorch_lightning.callbacks import Callback
from statistics import mean


class MinLossLogger(Callback):
    def __init__(self):
        # best loss epoch wise
        self.best_epoch_loss = None
        # the cumulative loss and the number of observations for making the epoch loss
        self.observation_cumulative_batch_loss = 0
        self.val_observations = 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.observation_cumulative_batch_loss += outputs * batch.shape[0]
        self.val_observations += batch.shape[0]

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch_loss = self.observation_cumulative_batch_loss / self.val_observations
        self.best_epoch_loss = epoch_loss if self.best_epoch_loss is None else min(self.best_epoch_loss, epoch_loss)

        # since an new epoch start both to 0
        self.observation_cumulative_batch_loss = 0
        self.val_observations = 0

    def on_train_end(self, trainer, pl_module):
        trainer.logger.log_metrics({"best_val_loss": self.best_epoch_loss})
