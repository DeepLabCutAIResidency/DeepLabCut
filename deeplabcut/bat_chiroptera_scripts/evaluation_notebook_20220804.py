"""
This notebook is written to get the test error for a specific subset of the test data
since Deeplabcut considers any frame not in the trainig dataset to be part of the test
dataset. Since I trained models on different subsets of the labeled data but wish to test
on a fixed set I need to hack this a bit.
"""

# %%
from getErrorDistribution import getErrorDistribution
import numpy as np

#%%
config_path = "/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-08-03/config.yaml"
modelprefix = "data_augm_00_none"

# %%
import pandas as pd
df = pd.read_hdf('/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-08-03/training-datasets/iteration-0/UnaugmentedDataSet_geneva_protocol_paper_austin_2020_bat_dataAug3/CollectedData_DLC.h5')
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
mean = []
devi = []
meanPCut = []
deviPcut = []
for snapshot in [0, 1, 2]:
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
for snapshot in [0, 1, 2]:
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
for snapshot in [0, 1, 2]:
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
for snapshot in [0, 1, 2]:
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

# %% A, shuffle 5
shuffle = 5
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
mean = []
devi = []
meanPCut = []
deviPcut = []
for snapshot in [0, 1, 2]:
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

# %% A+ref, shuffle 6
shuffle = 6
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
for snapshot in [0, 1, 2]:
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

# %% A+B, shuffle 7
shuffle = 7
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
for snapshot in [0, 1, 2]:
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

# %% A+B+25, shuffle 8
shuffle = 8
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
for snapshot in [0, 1, 2]:
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

# %% A, shuffle 9
shuffle = 9
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
mean = []
devi = []
meanPCut = []
deviPcut = []
for snapshot in [0, 1, 2]:
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

# %% A+ref, shuffle 10
shuffle = 10
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
for snapshot in [0, 1, 2]:
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

# %% A+B, shuffle 11
shuffle = 11
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
for snapshot in [0, 1, 2]:
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

# %% A+B+25, shuffle 12
shuffle = 12
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
for snapshot in [0, 1, 2]:
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