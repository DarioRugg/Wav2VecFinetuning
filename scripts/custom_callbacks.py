from pytorch_lightning.callbacks import Callback
from statistics import mean


class MinLossLogger(Callback):
    def __init__(self):
        # best loss epoch wise
        self.best_epoch_loss = None
        # the cumulative loss and the number of observations for making the epoch loss
        self.observation_cumulative_batch_loss = None
        self.val_observations = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        batch_size = batch[0].shape[0]
        self.observation_cumulative_batch_loss += outputs * batch_size
        self.val_observations += batch_size

    def on_validation_epoch_start(self, trainer, pl_module):
        # since an new epoch starts, set both to 0
        self.observation_cumulative_batch_loss = 0
        self.val_observations = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch_loss = self.observation_cumulative_batch_loss / self.val_observations
        self.best_epoch_loss = epoch_loss if self.best_epoch_loss is None else min(self.best_epoch_loss, epoch_loss)

    def on_train_end(self, trainer, pl_module):
        trainer.logger.log_metrics({"best_val_loss": self.best_epoch_loss})


class CustomEarlyStopping(MinLossLogger):
    def __init__(self,patience=3, tolerance=0, start_monitoring=0):
        super(CustomEarlyStopping, self).__init__()

        self.tolerance = tolerance
        self.patience = patience
        self.start_monitoring = start_monitoring

        self.waiting = None
        self.stop = False

    def on_validation_epoch_end(self, trainer, pl_module):
        super(CustomEarlyStopping, self).on_validation_epoch_end(trainer, pl_module)

        epoch_loss = self.observation_cumulative_batch_loss / self.val_observations
        if not self.stop and epoch_loss > self.best_epoch_loss + self.tolerance:
            self.waiting += 1
            if self.waiting >= self.patience:
                self.stop = True
        else:
            self.waiting = 0

    def on_train_start(self, trainer, pl_module):
        super(CustomEarlyStopping, self).on_train_start(trainer, pl_module)
        self.waiting = 0
        self.stop = False

    def on_train_end(self, trainer, pl_module):
        pass

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if self.stop: return -1


class ChartsLogger(Callback):
    def __init__(self, classes):
        self.classes = classes
        self.y = []
        self.y_hat = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.y += outputs["y"]
        self.y_hat += outputs["y_hat"]

    def on_test_start(self, trainer, pl_module):
        self.y = []
        self.y_hat = []

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch_loss = self.observation_cumulative_batch_loss / self.val_observations
        self.best_epoch_loss = epoch_loss if self.best_epoch_loss is None else min(self.best_epoch_loss, epoch_loss)

        # since an new epoch start both to 0
        self.observation_cumulative_batch_loss = 0
        self.val_observations = 0

    def on_test_end(self, trainer, pl_module):
        trainer.logger.log_metrics({"best_val_loss": self.best_epoch_loss})
