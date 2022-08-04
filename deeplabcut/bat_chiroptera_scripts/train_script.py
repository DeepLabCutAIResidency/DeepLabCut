import os
import deeplabcut
config_path = '/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-29/config.yaml'
shuffles = [1,2,3,4]
modelprefix = "data_augm_00_baseline"
max_snapshots_to_keep=20
displayiters=1000
maxiters=100000
saveiters=50000
gputouse=3
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
for shuffle in shuffles:
    deeplabcut.train_network(config_path, shuffle=shuffle, displayiters=displayiters, saveiters=saveiters, maxiters=maxiters, max_snapshots_to_keep=max_snapshots_to_keep, modelprefix=modelprefix, gputouse=gputouse, allow_growth=True)