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
All the models and datasets are inside the Assets folder
The structure of the folder is the following:

* Assets
  * Data
  * Models
  * Logs
  * Configs


The folders Data and Models are not in the repository since them are too heavy.

for the home test the directory "home_test" inside Logs/ is excluded from git