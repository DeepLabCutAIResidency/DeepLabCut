import os
import deeplabcut
config_path='/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-29/config.yaml'
model_prefix = 'data_augm_00_none'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gputouse=2
shuffles = [1,2,3,4]
for shuffle in shuffles:
    deeplabcut.train_network(
        config_path,
        shuffle=shuffle,
        modelprefix=model_prefix,
        gputouse=gputouse,
        allow_growth=True,
        trainingsetindex=shuffle-1,
        max_snapshots_to_keep=10,
        maxiters=150000
    )