from scripts.train_test import train, test


if __name__ == '__main__':
    config = "wav2vec_server.json"

    train(conf_file=config)
    test(conf_file=config)