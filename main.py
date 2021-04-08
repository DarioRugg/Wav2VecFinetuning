import hydra
from omegaconf import DictConfig
from scripts.train_test import train, test
from scripts.utils import server_setup

from torch.utils.tensorboard import SummaryWriter
import wandb


@hydra.main(config_path=r"Assets\Config", config_name="config.yaml")
def main(cfg: DictConfig):

    server_setup(cfg)

    logs_writer = SummaryWriter("TensorBoard_logs")
    wandb.init(project=cfg.simulation_name, dir=hydra.utils.get_original_cwd())

    train(cfg=cfg, tensorboard_writer=logs_writer)
    test(cfg=cfg, tensorboard_writer=logs_writer)

    logs_writer.close()

if __name__ == '__main__':
    main()