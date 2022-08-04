#%%%

from getErrorDistribution import getErrorDistribution
(
    ErrorDistribution_all,
    ErrorDistribution_test,
    ErrorDistribution_train,
    ErrorDistributionPCutOff_all,
    ErrorDistributionPCutOff_test,
    ErrorDistributionPCutOff_train
)  = getErrorDistribution(
        '/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-19/config.yaml',
        shuffle=1,
        snapindex=0,
        trainFractionIndex = 0,
        modelprefix = "data_augm_00_baseline"
)
#%%
(
    ErrorDistribution_all2,
    ErrorDistribution_test2,
    ErrorDistribution_train2,
    ErrorDistributionPCutOff_all2,
    ErrorDistributionPCutOff_test2,
    ErrorDistributionPCutOff_train2
)  = getErrorDistribution(
        '/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-19/config.yaml',
        shuffle=2,
        snapindex=0,
        trainFractionIndex = 0,
        modelprefix = "data_augm_00_baseline"
)
# %%
from scipy.stats import ranksums
ranksums(ErrorDistributionPCutOff_test2.values.flatten(), ErrorDistributionPCutOff_test.values.flatten(), nan_policy='omit')

# %%
import numpy as np
print(np.nanmean(ErrorDistribution_test2.values.flatten()))
print(np.nanmean(ErrorDistribution_test.values.flatten()))

# %%
import scipy.io
human_pixel_error = scipy.io.loadmat('/home/jonas2/DLC_files/Austin2020_for_protocol_paper/point_pixel_distance_human_digitizers.mat')
print(np.nanmean(human_pixel_error['euclidian_pixel_distance'].flatten()))
# %%
ranksums(ErrorDistributionPCutOff_test2.values.flatten(), human_pixel_error['euclidian_pixel_distance'].flatten(), nan_policy='omit',alternative='two-sided')

# %%
help(ranksums)
# %%
# %%
# %% ResNet50
snapindices=range(4)
shuffles=range(1,4)
errors=[]
for snapindex in snapindices:
    errorsTEMP=[]
    for shuffle in shuffles:
        (
            _,
            ErrorDistribution_test,
            _,
            _,
            ErrorDistributionPCutOff_test,
            _
        )  = getErrorDistribution(
            '/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-19/config.yaml',
            shuffle=shuffle,
            snapindex=snapindex,
            trainFractionIndex = 0,
            modelprefix = "data_augm_00_baseline"
        )
        errorsTEMP.append(ErrorDistribution_test)
    errors.append(np.concatenate((errorsTEMP), axis=0))
#%%
mean_and_standardError=np.empty((2,4))
for snapindex in snapindices:
    mean_and_standardError[0,snapindex]=np.nanmean(errors[snapindex].flatten())
    mean_and_standardError[1,snapindex]=np.nanstd(errors[snapindex].flatten())/(errors[snapindex].size**.5)
print(mean_and_standardError)

# %% ResNet101
snapindices=range(4)
shuffles=range(4,7)

errors=[]
for snapindex in snapindices:
    errorsTEMP=[]
    for shuffle in shuffles:
        (
            _,
            ErrorDistribution_test,
            _,
            _,
            ErrorDistributionPCutOff_test,
            _
        )  = getErrorDistribution(
            '/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-19/config.yaml',
            shuffle=shuffle,
            snapindex=snapindex,
            trainFractionIndex = 0,
            modelprefix = "data_augm_00_baseline"
        )
        errorsTEMP.append(ErrorDistribution_test)
    errors.append(np.concatenate((errorsTEMP), axis=0))
#%%
mean_and_standardError=np.empty((2,4))
for snapindex in snapindices:
    mean_and_standardError[0,snapindex]=np.nanmean(errors[snapindex].flatten())
    mean_and_standardError[1,snapindex]=np.nanstd(errors[snapindex].flatten())/(errors[snapindex].size**.5)
print(mean_and_standardError)

# %% ResNet152
snapindices=range(4)
shuffles=range(7,9)

errors=[]
for snapindex in snapindices:
    errorsTEMP=[]
    for shuffle in shuffles:
        (
            _,
            ErrorDistribution_test,
            _,
            _,
            ErrorDistributionPCutOff_test,
            _
        )  = getErrorDistribution(
            '/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-19/config.yaml',
            shuffle=shuffle,
            snapindex=snapindex,
            trainFractionIndex = 0,
            modelprefix = "data_augm_00_baseline"
        )
        errorsTEMP.append(ErrorDistribution_test)
    errors.append(np.concatenate((errorsTEMP), axis=0))
#%%
mean_and_standardError=np.empty((2,4))
for snapindex in snapindices:
    mean_and_standardError[0,snapindex]=np.nanmean(errors[snapindex].flatten())
    mean_and_standardError[1,snapindex]=np.nanstd(errors[snapindex].flatten())/(errors[snapindex].size**.5)
print(mean_and_standardError)

# %%
(
    ErrorDistribution_all,
    ErrorDistribution_test,
    ErrorDistribution_train,
    ErrorDistributionPCutOff_all,
    ErrorDistributionPCutOff_test,
    ErrorDistributionPCutOff_train
)  = getErrorDistribution(
        '/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-19/config.yaml',
        shuffle=4,
        snapindex=0,
        trainFractionIndex = 0,
        modelprefix = "data_augm_01_fliplr_gb_sc_rot_bright"
)
# %%
np.nanmean(ErrorDistribution_test.values.flatten())
np.nanstd(ErrorDistribution_test.values.flatten())/(ErrorDistribution_test.size**.5)
# %%
