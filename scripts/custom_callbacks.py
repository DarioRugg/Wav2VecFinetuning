from pytorch_lightning.callbacks import Callback
import wandb


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


class ChartsLogger(Callback):
    def __init__(self, classes):
        self.classes = classes
        self.y = None
        self.y_hat = None

    def on_test_start(self, trainer, pl_module):
        self.y = []
        self.y_hat = []
        print(" ---> start y and y_hat len 0")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.y += outputs["y"]
        self.y_hat += outputs["y_hat"]

        print(f" ---- Batch {batch_idx} ---- \n  len y: {len(outputs['y'])}, len y_hat: {len(outputs['y_hat'])}"
              f"\n  len y: {len(self.y)}, len y_hat: {len(self.y_hat)}")
        if batch_idx == 0: print(f" -  y_hat example: {type(outputs['y_hat'])}{outputs['y_hat']}")
        if batch_idx == 0: print(f" -  y     example: {type(outputs['y'])}{outputs['y']}")

    def on_test_end(self, trainer, pl_module):
        print(f"------------ End ------------ \n        len y: {len(self.y)}, len y_hat: {len(self.y_hat)}")
        wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                           y_true=self.y,
                                                           preds=self.y_hat,
                                                           class_names=self.classes)})

        wandb.log({"roc_curve": wandb.plot.roc_curve(self.y, self.y_hat, labels=self.classes)})
