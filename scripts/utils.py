from scripts.classification_models import SpectrogramCNN, EfficientNetModel
from scripts.wav2vec_models import Wav2VecComplete, Wav2VecFeatureExtractorGAP, Wav2VecFeezingEncoderOnly, \
    Wav2VecCLSToken, Wav2VecCLSTokenNotPretrained, Wav2VecFeatureExtractor, Wav2VecCLSPaperFinetuning


def get_defaults_hyperparameters(hydra_cfg):
    return dict(
        training_batches=hydra_cfg.machine.training_batches,
        epochs=hydra_cfg.model.epochs,
        learning_rate=hydra_cfg.model.learning_rate,
        classifier_hidden_layers=hydra_cfg.model.hidden_layers,
        classifier_hidden_size=hydra_cfg.model.hidden_size
    )


def update_sweep_configs(hydra_cfg, sweep_cfg):
    hydra_cfg.machine.training_batches = sweep_cfg["training_batches"]
    hydra_cfg.model.epochs = sweep_cfg["epochs"]
    hydra_cfg.model.learning_rate = sweep_cfg["learning_rate"]
    hydra_cfg.model.hidden_layers = sweep_cfg["classifier_hidden_layers"]
    hydra_cfg.model.hidden_size = sweep_cfg["classifier_hidden_size"]


def get_model(cfg):
    if cfg.model.name.lower() == "cnn":
        return SpectrogramCNN(input_size=cfg.model.input_size, class_number=cfg.dataset.number_of_classes)
    elif cfg.model.name.lower() == "efficientnet":
        return EfficientNetModel(num_classes=cfg.dataset.number_of_classes, blocks=cfg.model.blocks,
                                 learning_rate=cfg.model.learning_rate)
    elif cfg.model.name.lower() == "wav2vec":
        if cfg.model.option == "partial":
            return Wav2VecFeezingEncoderOnly(num_classes=cfg.dataset.number_of_classes)
        elif cfg.model.option == "all":
            return Wav2VecComplete(num_classes=cfg.dataset.number_of_classes, finetune_pretrained=cfg.model.finetuning)
        elif cfg.model.option == "cnn":
            return Wav2VecFeatureExtractor(num_classes=cfg.dataset.number_of_classes,
                                           finetune_pretrained=cfg.model.finetuning)
        elif cfg.model.option == "cnn_avg":
            return Wav2VecFeatureExtractorGAP(num_classes=cfg.dataset.number_of_classes,
                                              finetune_pretrained=cfg.model.finetuning)
        elif cfg.model.option == "cls_token":
            return Wav2VecCLSToken(num_classes=cfg.dataset.number_of_classes)
        elif cfg.model.option == "cls_token_not_pretrained":
            return Wav2VecCLSTokenNotPretrained(num_classes=cfg.dataset.number_of_classes)
        elif cfg.model.option == "paper":
            return Wav2VecCLSPaperFinetuning(num_classes=cfg.dataset.number_of_classes,
                                             learning_rate=cfg.model.learning_rate,
                                             num_epochs=cfg.model.epochs,
                                             hidden_layers=cfg.model.hidden_layers,
                                             hidden_size=cfg.model.hidden_size)


def get_model_from_checkpoint(cfg, checkpoint_path):
    if cfg.model.name.lower() == "cnn":
        return SpectrogramCNN.load_from_checkpoint(checkpoint_path, input_size=cfg.model.input_size,
                                                   class_number=cfg.dataset.number_of_classes)
    elif cfg.model.name.lower() == "efficientnet":
        return EfficientNetModel.load_from_checkpoint(checkpoint_path, num_classes=cfg.dataset.number_of_classes,
                                                      blocks=cfg.model.blocks, learning_rate=cfg.model.learning_rate)
    elif cfg.model.name.lower() == "wav2vec":
        if cfg.model.option == "partial":
            return Wav2VecFeezingEncoderOnly.load_from_checkpoint(checkpoint_path,
                                                                  num_classes=cfg.dataset.number_of_classes)
        elif cfg.model.option == "all":
            return Wav2VecComplete.load_from_checkpoint(checkpoint_path, num_classes=cfg.dataset.number_of_classes,
                                                        finetune_pretrained=cfg.model.finetuning)
        elif cfg.model.option == "cnn":
            return Wav2VecFeatureExtractor.load_from_checkpoint(checkpoint_path,
                                                                num_classes=cfg.dataset.number_of_classes,
                                                                finetune_pretrained=cfg.model.finetuning)
        elif cfg.model.option == "cnn_avg":
            return Wav2VecFeatureExtractorGAP.load_from_checkpoint(checkpoint_path,
                                                                   num_classes=cfg.dataset.number_of_classes,
                                                                   finetune_pretrained=cfg.model.finetuning)
        elif cfg.model.option == "cls_token":
            return Wav2VecCLSToken.load_from_checkpoint(checkpoint_path, num_classes=cfg.dataset.number_of_classes)
        elif cfg.model.option == "cls_token_not_pretrained":
            return Wav2VecCLSTokenNotPretrained.load_from_checkpoint(checkpoint_path,
                                                                     num_classes=cfg.dataset.number_of_classes)
        elif cfg.model.option == "paper":
            return Wav2VecCLSPaperFinetuning.load_from_checkpoint(checkpoint_path,
                                                                  num_classes=cfg.dataset.number_of_classes,
                                                                  learning_rate=cfg.model.learning_rate,
                                                                  num_epochs=cfg.model.epochs,
                                                                  hidden_layers=cfg.model.hidden_layers,
                                                                  hidden_size=cfg.model.hidden_size)
    else:
        raise (cfg.model.name, "not integrated with pytorch lightening yet!")
