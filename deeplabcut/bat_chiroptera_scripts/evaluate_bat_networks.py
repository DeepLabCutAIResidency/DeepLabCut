#%%
import deeplabcut

#%%
#config_path = '/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-08-03/config.yaml'
config_path = '/home/jonas2/DLC_files/projects/bat_augmentation_austin_2020_bat_data-DLC-2022-08-18/config.yaml'
#model_prefix = 'data_augm_00_baseline'
#model_prefix = 'data_augm_00_none'
#model_prefix = 'data_augm_01_fliplr'
model_prefix = 'data_augm_03_max_rotate'


#%%
deeplabcut.evaluate_network(config_path, modelprefix = model_prefix, Shuffles = [4], gputouse=1, trainingsetindex=3, plotting=False)
# %%
help(deeplabcut.evaluate_network)
# %%
