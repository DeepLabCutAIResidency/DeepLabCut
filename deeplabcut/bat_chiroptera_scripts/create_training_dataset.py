# %%
from cgi import test
import deeplabcut
import pandas as pd
config_path = '/home/neslihan/foraging-F-2020-11-24/config.yaml'
# %%
df = pd.read_hdf('/home/neslihan/foraging-F-2020-11-24/data_augm_grayscale/training-datasets/iteration-0/UnaugmentedDataSet_foragingNov24/CollectedData_F.h5')
image_paths = df.index.to_list()
test_folder = {'104_60holes_ncover_cut_day2_covered_60_2',
 '104_60holes_ncover_cut_day5_2',
 '115_60holes_ncover_cut_day2_covered_60_2',
 '115_60holes_ncover_cut_day5_2',
 '554_60holes_ncover_cut_day2_covered_60_2',
 '554_60holes_ncover_cut_day5',
 '554_60holes_ncover_cut_day5_2',
 '229_60holes_ncover_cut_day5',
 '229_60holes_ycover_cut_day5',
 '395_60holes_ncover_cut_day2_covered_60_2',
 '395_60holes_ncover_cut_day5_2',
 '543_60holes_ncover_cut_day2_covered_60_2',
 '543_60holes_ncover_cut_day5_2',
 '868_60holes_ycover_reversedplatforms_05102020_cut_day5',
 '868_60holes_ycover_retested_02112020_cut_day5',
 '868_60holes_ncover_cut_day5',
 '857_60holes_ycover_cut_day5',
 '857_60holes_ncover_cut_day5',
 '802_60holes_ycover_cut_day5',
 '802_60holes_ncover_cut_day5'}
test_inds = []
for i, path in enumerate(image_paths):
    if path[1] in test_folder:
        test_inds.append(i)
train_inds = list(range(len(df)))
for ind in test_inds:
    train_inds.remove(ind)

train_inds = [j for j in range(len(df)) if j not in test_inds]

#train_inds = set(range(len(df))).difference(test_inds)
# %%
deeplabcut.create_training_dataset(
    config_path,
    trainIndices=[train_inds],
    testIndices=[test_inds]
)
# %%
import pickle
df2 = pd.read_pickle("/home/neslihan/foraging-F-2020-11-24/training-datasets/iteration-0/UnaugmentedDataSet_foragingNov24/Documentation_data-foraging_43shuffle1.pickle")
len(df2[2])
# %%
