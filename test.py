from scripts.wav2vec_models import Wav2VecComplete

model = Wav2VecComplete(8)

for child in model.pretrained_model.encoder.modules():
    print("  ---->    ", child)