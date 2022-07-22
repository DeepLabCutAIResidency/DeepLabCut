#%%
import deeplabcut

#%%
config_path = '/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-19/config.yaml'
model_prefix = 'data_augm_00_baseline'

#%%
deeplabcut.evaluate_network(config_path, modelprefix = model_prefix, Shuffles = [2], gputouse=3)
# %%
help(deeplabcut.evaluate_network)
# %%
