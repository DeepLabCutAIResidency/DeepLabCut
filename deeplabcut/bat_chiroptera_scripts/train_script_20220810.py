import os
import deeplabcut
config_path='/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-08-03/config.yaml'
model_prefix = 'data_augm_01_fliplr'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gputouse=2
shuffles = [14]
trainingsetindices = [1]
for shuffle, trainingsetindex in zip(shuffles, trainingsetindices):
    deeplabcut.train_network(
        config_path,
        shuffle=shuffle,
        modelprefix=model_prefix,
        gputouse=gputouse,
        allow_growth=True,
        trainingsetindex=trainingsetindex,
        max_snapshots_to_keep=20,
        saveiters=10000,
        maxiters=300000
    )