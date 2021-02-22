# Finetuning repository for the Wav2Vec XLSR model 

finetuning on the DEMoS dataset.

## To run:
To fine-tune the classification model you need to run the finetuning.py file.

for run the finetuning in a persistent terminal (tmux):
```bash
tmux new -s dario_session
conda activate dario_env
python finetuning.py
```

## Assets and data:
All the models and datasets are inside the Assets folder (excluded from the git repository)
The structure of the folder is the following:

* Assets
  * Data
  * Models

the *Logs* folder is instead outside of *Assets* so that is included on the git repository
