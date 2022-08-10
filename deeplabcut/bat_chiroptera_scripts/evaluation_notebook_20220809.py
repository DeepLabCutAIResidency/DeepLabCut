"""
This notebook is written to get the test error for a specific subset of the test data
since Deeplabcut considers any frame not in the trainig dataset to be part of the test
dataset. Since I trained models on different subsets of the labeled data but wish to test
on a fixed set I need to hack this a bit.
"""

# %%
from getErrorDistribution import getErrorDistribution
import numpy as np
import deeplabcut
import matplotlib.pyplot as plt

#%%
config_path = "/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-08-03/config.yaml"
modelprefix = "data_augm_00_none"

# %%
import pandas as pd
df = pd.read_hdf('/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-08-03/training-datasets/iteration-0/UnaugmentedDataSet_geneva_protocol_paper_austin_2020_bat_dataAug3/CollectedData_DLC.h5')
image_paths = df.index.to_list()


# %%
test_inds = []

for i, path in enumerate(image_paths):
    if str(path[1]).endswith("_50_test"):
        test_inds.append(i)

test_paths = list(set([image_paths[i][1] for i in test_inds]))

#%% sorted output on spreadsheet:
test_paths_cam1 =   ['TS5-544-Cam1_2020-06-25_000099Track8_50_test',
                    'TS5-544-Cam1_2020-06-25_000103Track3_50_test',
                    'TS5-544-Cam1_2020-06-25_000104Track3_50_test',
                    'TS5-544-Cam1_2020-06-25_000108Track6_50_test',
                    'TS5-544-Cam1_2020-06-25_000123Track6_50_test',
                    'TS5-544-Cam1_2020-06-25_000128Track2_50_test',
                    'TS5-544-Cam1_2020-06-25_000134Track5_50_test'
                    ]
test_paths_cam2 =   ['IL5-519-Cam2_2020-06-25_000099Track6_50_test',
                    'IL5-519-Cam2_2020-06-25_000103Track3_50_test',
                    'IL5-519-Cam2_2020-06-25_000104Track2_50_test',
                    'IL5-519-Cam2_2020-06-25_000109Track1_50_test',
                    'IL5-519-Cam2_2020-06-25_000124Track9_50_test',
                    'IL5-519-Cam2_2020-06-25_000130Track2_50_test',
                    'IL5-519-Cam2_2020-06-25_000136Track10_50_test'
                    ]
test_paths_cam3 =   ['IL5-534-Cam3_2020-06-25_000095Track14_50_test',
                    'IL5-534-Cam3_2020-06-25_000100Track4_50_test',
                    'IL5-534-Cam3_2020-06-25_000101Track4_50_test',
                    'IL5-534-Cam3_2020-06-25_000106Track3_50_test',
                    'IL5-534-Cam3_2020-06-25_000122Track7_50_test',
                    'IL5-534-Cam3_2020-06-25_000127Track4_50_test',
                    'IL5-534-Cam3_2020-06-25_000133Track9_50_test'
                    ]

test_inds_cam1 = [[],[],[],[],[],[],[]]
test_inds_cam2 = [[],[],[],[],[],[],[]]
test_inds_cam3 = [[],[],[],[],[],[],[]]

#%%
for i, path in enumerate(image_paths):
    for j in range(7):
        if str(path[1]).__eq__(test_paths_cam1[j]):
            test_inds_cam1[j].append(i)
        elif str(path[1]).__eq__(test_paths_cam2[j]):
            test_inds_cam2[j].append(i)
        elif str(path[1]).__eq__(test_paths_cam3[j]):
            test_inds_cam3[j].append(i)

# %%
mean_cam1 = np.zeros([12,7]) # shuffle x movie
mean_cam2 = np.zeros([12,7])
mean_cam3 = np.zeros([12,7])

devi_cam1 = np.zeros([12,7])
devi_cam2 = np.zeros([12,7])
devi_cam3 = np.zeros([12,7])

meanPcut_cam1 = np.zeros([12,7]) # shuffle x movie
meanPcut_cam2 = np.zeros([12,7])
meanPcut_cam3 = np.zeros([12,7])

deviPcut_cam1 = np.zeros([12,7])
deviPcut_cam2 = np.zeros([12,7])
deviPcut_cam3 = np.zeros([12,7])

# %%
snapshot = 1 #100k iterations
for shuffle in range(1,13):
    trainFractionIndex = np.mod(shuffle-1,4)
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
    for movie_number in range(7):
        mean_cam1[shuffle-1,movie_number] = np.nanmean(ErrorDistribution_all.values[test_inds_cam1[movie_number]][:])
        devi_cam1[shuffle-1,movie_number] = np.nanstd(ErrorDistribution_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds_cam1[movie_number]][:].size**.5)

        meanPcut_cam1[shuffle-1,movie_number] = np.nanmean(ErrorDistributionPCutOff_all.values[test_inds_cam1[movie_number]][:])
        deviPcut_cam1[shuffle-1,movie_number] = np.nanstd(ErrorDistributionPCutOff_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds_cam1[movie_number]][:].size**.5)

        mean_cam2[shuffle-1,movie_number] = np.nanmean(ErrorDistribution_all.values[test_inds_cam2[movie_number]][:])
        devi_cam2[shuffle-1,movie_number] = np.nanstd(ErrorDistribution_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds_cam2[movie_number]][:].size**.5)

        meanPcut_cam2[shuffle-1,movie_number] = np.nanmean(ErrorDistributionPCutOff_all.values[test_inds_cam2[movie_number]][:])
        deviPcut_cam2[shuffle-1,movie_number] = np.nanstd(ErrorDistributionPCutOff_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds_cam2[movie_number]][:].size**.5)

        mean_cam3[shuffle-1,movie_number] = np.nanmean(ErrorDistribution_all.values[test_inds_cam3[movie_number]][:])
        devi_cam3[shuffle-1,movie_number] = np.nanstd(ErrorDistribution_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds_cam3[movie_number]][:].size**.5)

        meanPcut_cam3[shuffle-1,movie_number] = np.nanmean(ErrorDistributionPCutOff_all.values[test_inds_cam3[movie_number]][:])
        deviPcut_cam3[shuffle-1,movie_number] = np.nanstd(ErrorDistributionPCutOff_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds_cam3[movie_number]][:].size**.5)

# %%

for shuffle in range(12):
    if np.mod(shuffle,4) == 0:
        linestyle = '-'
    elif np.mod(shuffle,4) == 1:
        linestyle = '--'
    elif np.mod(shuffle,4) == 2:
        linestyle = '-.'
    elif np.mod(shuffle,4) == 3:
        linestyle = ':'
    
    if 0 <= shuffle <= 3:
        color = (0, 0.4470, 0.7410)
    elif 4 <= shuffle <= 7:
        color = (0.8500, 0.3250, 0.0980)
    elif 8 <= shuffle <= 11:
        color = (0.9290, 0.6940, 0.1250)

    movie_number = list(range(1,8))
    movie_number = [x - 6/100 + shuffle/100 for x in movie_number]
    
    plt.rcParams['figure.figsize'] = [15, 10]
    plt.subplot(3,1,1)
    plt.errorbar(movie_number,meanPcut_cam1[shuffle,:], deviPcut_cam1[shuffle,:,], linestyle=linestyle, color = color)
    #plt.legend(["Half",'Half+ref',"Full", "Full+ref","Half IS",'Half+ref IS',"Full IS", "Full+ref IS", "Half MS",'Half+ref MS',"Full MS", "Full+ref MS"])
    plt.ylim([0, 50])
    plt.yticks([0, 5, 10, 20, 30, 40, 50])

    plt.subplot(3,1,2)
    plt.errorbar(movie_number,meanPcut_cam2[shuffle,:], deviPcut_cam2[shuffle,:,], linestyle=linestyle, color = color)
    plt.ylim([0, 50])
    plt.yticks([0, 5, 10, 20, 30, 40, 50])
    plt.subplot(3,1,3)
    plt.errorbar(movie_number,meanPcut_cam3[shuffle,:], deviPcut_cam3[shuffle,:,], linestyle=linestyle, color = color)
    plt.ylim([0, 50])
    plt.yticks([0, 5, 10, 20, 30, 40, 50])
    #plt.legend(["Half",'Half+ref',"Full", "Full+ref","Half IS",'Half+ref IS',"Full IS", "Full+ref IS", "Half MS",'Half+ref MS',"Full MS", "Full+ref MS"])

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
    devi.append(np.nanstd(ErrorDistribution_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds][:].size**.5))
    
    meanPCut.append(np.nanmean(ErrorDistributionPCutOff_all.values[test_inds][:]))
    deviPcut.append(np.nanstd(ErrorDistributionPCutOff_all.values[test_inds][:])/(ErrorDistributionPCutOff_all.values[test_inds][:].size**.5))  


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
    devi.append(np.nanstd(ErrorDistribution_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds][:].size**.5))
    
    meanPCut.append(np.nanmean(ErrorDistributionPCutOff_all.values[test_inds][:]))
    deviPcut.append(np.nanstd(ErrorDistributionPCutOff_all.values[test_inds][:])/(ErrorDistributionPCutOff_all.values[test_inds][:].size**.5))

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
    devi.append(np.nanstd(ErrorDistribution_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds][:].size**.5))
    
    meanPCut.append(np.nanmean(ErrorDistributionPCutOff_all.values[test_inds][:]))
    deviPcut.append(np.nanstd(ErrorDistributionPCutOff_all.values[test_inds][:])/(ErrorDistributionPCutOff_all.values[test_inds][:].size**.5))

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
    devi.append(np.nanstd(ErrorDistribution_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds][:].size**.5))
    
    meanPCut.append(np.nanmean(ErrorDistributionPCutOff_all.values[test_inds][:]))
    deviPcut.append(np.nanstd(ErrorDistributionPCutOff_all.values[test_inds][:])/(ErrorDistributionPCutOff_all.values[test_inds][:].size**.5))
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
    devi.append(np.nanstd(ErrorDistribution_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds][:].size**.5))
    
    meanPCut.append(np.nanmean(ErrorDistributionPCutOff_all.values[test_inds][:]))
    deviPcut.append(np.nanstd(ErrorDistributionPCutOff_all.values[test_inds][:])/(ErrorDistributionPCutOff_all.values[test_inds][:].size**.5))  

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
    devi.append(np.nanstd(ErrorDistribution_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds][:].size**.5))
    
    meanPCut.append(np.nanmean(ErrorDistributionPCutOff_all.values[test_inds][:]))
    deviPcut.append(np.nanstd(ErrorDistributionPCutOff_all.values[test_inds][:])/(ErrorDistributionPCutOff_all.values[test_inds][:].size**.5))

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
    devi.append(np.nanstd(ErrorDistribution_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds][:].size**.5))
    
    meanPCut.append(np.nanmean(ErrorDistributionPCutOff_all.values[test_inds][:]))
    deviPcut.append(np.nanstd(ErrorDistributionPCutOff_all.values[test_inds][:])/(ErrorDistributionPCutOff_all.values[test_inds][:].size**.5))

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
    devi.append(np.nanstd(ErrorDistribution_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds][:].size**.5))
    
    meanPCut.append(np.nanmean(ErrorDistributionPCutOff_all.values[test_inds][:]))
    deviPcut.append(np.nanstd(ErrorDistributionPCutOff_all.values[test_inds][:])/(ErrorDistributionPCutOff_all.values[test_inds][:].size**.5))
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
    devi.append(np.nanstd(ErrorDistribution_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds][:].size**.5))
    
    meanPCut.append(np.nanmean(ErrorDistributionPCutOff_all.values[test_inds][:]))
    deviPcut.append(np.nanstd(ErrorDistributionPCutOff_all.values[test_inds][:])/(ErrorDistributionPCutOff_all.values[test_inds][:].size**.5))  

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
    devi.append(np.nanstd(ErrorDistribution_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds][:].size**.5))
    
    meanPCut.append(np.nanmean(ErrorDistributionPCutOff_all.values[test_inds][:]))
    deviPcut.append(np.nanstd(ErrorDistributionPCutOff_all.values[test_inds][:])/(ErrorDistributionPCutOff_all.values[test_inds][:].size**.5))

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
    devi.append(np.nanstd(ErrorDistribution_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds][:].size**.5))

    meanPCut.append(np.nanmean(ErrorDistributionPCutOff_all.values[test_inds][:]))
    deviPcut.append(np.nanstd(ErrorDistributionPCutOff_all.values[test_inds][:])/(ErrorDistributionPCutOff_all.values[test_inds][:].size**.5))

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
    devi.append(np.nanstd(ErrorDistribution_all.values[test_inds][:])/(ErrorDistribution_all.values[test_inds][:].size**.5))
    
    meanPCut.append(np.nanmean(ErrorDistributionPCutOff_all.values[test_inds][:]))
    deviPcut.append(np.nanstd(ErrorDistributionPCutOff_all.values[test_inds][:])/(ErrorDistributionPCutOff_all.values[test_inds][:].size**.5))
# %%