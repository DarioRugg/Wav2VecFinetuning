from scripts.librosa_dataloaders import DEMoSDataset

dataset = dataset = DEMoSDataset(root_dir="Assets\Data\DEMoS_dataset_short_test", padding_cropping_size=10000, spectrogram=True, sampling_rate=None)

print(dataset[0][0].shape)