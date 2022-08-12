"""
This notebook is written to get the test error for a specific subset of the test data
since Deeplabcut considers any frame not in the trainig dataset to be part of the test
dataset. Since I trained models on different subsets of the labeled data but wish to test
on a fixed set I need to hack this a bit.
"""

#%%
config_path = "/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-29/config.yaml"
modelprefix = "data_augm_00_none"

# %%
import pandas as pd
df = pd.read_hdf('/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-29/training-datasets/iteration-0/UnaugmentedDataSet_geneva_protocol_paper_austin_2020_bat_dataJul29/CollectedData_DLC.h5')
image_paths = df.index.to_list()

# %% A, shuffle 1
shuffle = 1
trainFractionIndex = 0
# %%
test_inds = []
train_inds = []
for i, path in enumerate(image_paths):
    if str(path[1]).endswith("_A"):
        train_inds.append(i)
    elif str(path[1]).endswith("_50_test"):
        test_inds.append(i)

# %%
from getErrorDistribution import getErrorDistribution
import numpy as np
mean = []
devi = []
meanPCut = []
deviPcut = []
for snapshot in [0, 1, 2, 3, 4, 5, 6, 7]:
    (
        ErrorDistribution_all,
        _,
        _,
        ErrorDistributionPCutOff_all,
        _,
        _
    )  = getErrorDistribution(
        config_path,
        shuffle=shuffle,
        snapindex=snapshot,
        trainFractionIndex = trainFractionIndex,
        modelprefix = modelprefix
    )
    mean.append(np.nanmean(ErrorDistribution_all.values[test_inds][:]))
    devi.append(np.nanmean(ErrorDistribution_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds][:].size**.5))
    
    meanPCut.append(np.nanmean(ErrorDistributionPCutOff_all.values[test_inds][:]))
    deviPcut.append(np.nanmean(ErrorDistributionPCutOff_all.values[test_inds][:])/(ErrorDistributionPCutOff_all.values[test_inds][:].size**.5))  

# %% A+ref, shuffle 2
shuffle = 2
trainFractionIndex = 1

# %%
test_inds = []
train_inds = []
for i, path in enumerate(image_paths):
    if str(path[1]).endswith("_A"):
        train_inds.append(i)
    elif str(path[1]).endswith("_25_test"):
        train_inds.append(i)
    elif str(path[1]).endswith("_50_test"):
        test_inds.append(i)
# %%
mean = []
devi = []
meanPCut = []
deviPcut = []
for snapshot in [0, 1, 2, 3, 4, 5, 6, 7]:
    (
        ErrorDistribution_all,
        _,
        _,
        ErrorDistributionPCutOff_all,
        _,
        _
    )  = getErrorDistribution(
        config_path,
        shuffle=shuffle,
        snapindex=snapshot,
        trainFractionIndex = trainFractionIndex,
        modelprefix = modelprefix
    )
    mean.append(np.nanmean(ErrorDistribution_all.values[test_inds][:]))
    devi.append(np.nanmean(ErrorDistribution_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds][:].size**.5))
    
    meanPCut.append(np.nanmean(ErrorDistributionPCutOff_all.values[test_inds][:]))
    deviPcut.append(np.nanmean(ErrorDistributionPCutOff_all.values[test_inds][:])/(ErrorDistributionPCutOff_all.values[test_inds][:].size**.5))

# %% A+B, shuffle 3
shuffle = 3
trainFractionIndex = 2

# %%
test_inds = []
train_inds = []
test_inds = []
train_inds = []
for i, path in enumerate(image_paths):
    if str(path[1]).endswith("_A"):
        train_inds.append(i)
    elif str(path[1]).endswith("_B"):
        train_inds.append(i)
    elif str(path[1]).endswith("_50_test"):
        test_inds.append(i)
# %%
mean = []
devi = []
meanPCut = []
deviPcut = []
for snapshot in [0, 1, 2, 3, 4, 5, 6, 7]:
    (
        ErrorDistribution_all,
        _,
        _,
        ErrorDistributionPCutOff_all,
        _,
        _
    )  = getErrorDistribution(
        config_path,
        shuffle=shuffle,
        snapindex=snapshot,
        trainFractionIndex = trainFractionIndex,
        modelprefix = modelprefix
    )
    mean.append(np.nanmean(ErrorDistribution_all.values[test_inds][:]))
    devi.append(np.nanmean(ErrorDistribution_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds][:].size**.5))
    
    meanPCut.append(np.nanmean(ErrorDistributionPCutOff_all.values[test_inds][:]))
    deviPcut.append(np.nanmean(ErrorDistributionPCutOff_all.values[test_inds][:])/(ErrorDistributionPCutOff_all.values[test_inds][:].size**.5))

# %% A+B+25, shuffle 4
shuffle = 4
trainFractionIndex = 3
# %%
test_inds = []
train_inds = []
for i, path in enumerate(image_paths):
    if str(path[1]).endswith("_A"):
        train_inds.append(i)
    elif str(path[1]).endswith("_B"):
        train_inds.append(i)
    elif str(path[1]).endswith("_25_test"):
        train_inds.append(i)
    elif str(path[1]).endswith("_50_test"):
        test_inds.append(i)

# %%
mean = []
devi = []
meanPCut = []
deviPcut = []
for snapshot in [0, 1, 2, 3, 4]:
    (
        ErrorDistribution_all,
        _,
        _,
        ErrorDistributionPCutOff_all,
        _,
        _
    )  = getErrorDistribution(
        config_path,
        shuffle=shuffle,
        snapindex=snapshot,
        trainFractionIndex = trainFractionIndex,
        modelprefix = modelprefix
    )
    mean.append(np.nanmean(ErrorDistribution_all.values[test_inds][:]))
    devi.append(np.nanmean(ErrorDistribution_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds][:].size**.5))
    
    meanPCut.append(np.nanmean(ErrorDistributionPCutOff_all.values[test_inds][:]))
    deviPcut.append(np.nanmean(ErrorDistributionPCutOff_all.values[test_inds][:])/(ErrorDistributionPCutOff_all.values[test_inds][:].size**.5))
# %%
