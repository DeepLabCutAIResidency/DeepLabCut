#%%
import deeplabcut

#%%
config_path = '/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-29/config.yaml'
#model_prefix = 'data_augm_00_baseline'
model_prefix = 'data_augm_00_none'

#%%
deeplabcut.evaluate_network(config_path, modelprefix = model_prefix, Shuffles = [1], gputouse=3, trainingsetindex=0, plotting=False)
# %%
help(deeplabcut.evaluate_network)
# %%
