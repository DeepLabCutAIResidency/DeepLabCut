import os
import deeplabcut
config_path='/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-08-03/config.yaml'
model_prefix = 'data_augm_00_none'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gputouse=2
shuffles = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
trainingsetindices = [2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
for shuffle, trainingsetindex in zip(shuffles, trainingsetindices):
    deeplabcut.train_network(
        config_path,
        shuffle=shuffle,
        modelprefix=model_prefix,
        gputouse=gputouse,
        allow_growth=True,
        trainingsetindex=trainingsetindex,
        max_snapshots_to_keep=15,
    )