import hydra
from omegaconf import DictConfig
from scripts.train_test import train, test


@hydra.main(config_path=r"Assets\Configs_hydra", config_name="config.yaml")
def main(cfg: DictConfig):
    train(conf=cfg)
    test(conf=cfg)

if __name__ == '__main__':
    main()