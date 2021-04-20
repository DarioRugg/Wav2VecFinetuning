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


## running finetuning using Hydra from terminal
example of command:
python hydra_test_home.py machine=server model=efficientnet dataset=demos --cfg job

Where `model` must be specified and `machine=server` and `dataset=demos` have those configs as default.
The option `--cfg job` is instead for print the config at the beginning of the run.

## Reference to look at:
(colab notebook)[https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLSR_Wav2Vec2_on_Turkish_ASR_with_%F0%9F%A4%97_Transformers.ipynb#scrollTo=K_JUmf3G3b9S]
