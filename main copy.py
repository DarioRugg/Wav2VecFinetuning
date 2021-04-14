import hydra
from omegaconf import DictConfig
from scripts.train_test_lightning import train, test
from scripts.utils import server_setup

from torch.utils.tensorboard import SummaryWriter
import wandb


@hydra.main(config_path=r"Assets\Config", config_name="config.yaml")
def main(cfg: DictConfig):
    print(cfg.model.input_size, type(cfg.model.input_size))

if __name__ == '__main__':
    main()