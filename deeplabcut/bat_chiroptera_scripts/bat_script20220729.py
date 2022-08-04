#%%%
from sklearn.utils import shuffle
import deeplabcut
#%%%

#project_folder = "/home/jonas2/DLC_files/projects/"
#
#deeplabcut.create_new_project(
#            project='geneva_protocol_paper_austin_2020_bat_data',
#            experimenter='DLC',
#            videos=['/home/jonas2/DLC_files/projects/dummyVideos/'],
#            working_directory=project_folder
#        )

#%%%
config_path = "/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-29/config.yaml"
#%%%
deeplabcut.convertcsv2h5(config_path, scorer= 'DLC', userfeedback=False)

# %%
deeplabcut.check_labels(config_path)
# %%
# Dummy training dataset to get indexes and so on from later
deeplabcut.create_training_dataset(config_path, Shuffles=[99])

#%%
import pandas as pd
df = pd.read_hdf('/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-29/training-datasets/iteration-0/UnaugmentedDataSet_geneva_protocol_paper_austin_2020_bat_dataJul29/CollectedData_DLC.h5')
image_paths = df.index.to_list()


# %%
# train on A, shuffle 1
test_inds = []
train_inds = []
for i, path in enumerate(image_paths):
    if str(path[1]).endswith("_A"):
        train_inds.append(i)
    elif str(path[1]).endswith("_50_test"):
        test_inds.append(i)

deeplabcut.create_training_dataset(
    config_path,
    Shuffles=[1],
    trainIndices=[train_inds],
    testIndices=[test_inds],
    net_type="resnet_101",
    augmenter_type="imgaug"
)

# %%
# train on A+25, shuffle 2
test_inds = []
train_inds = []
for i, path in enumerate(image_paths):
    if str(path[1]).endswith("_A"):
        train_inds.append(i)
    elif str(path[1]).endswith("_25_test"):
        train_inds.append(i)
    elif str(path[1]).endswith("_50_test"):
        test_inds.append(i)

deeplabcut.create_training_dataset(
    config_path,
    Shuffles=[2],
    trainIndices=[train_inds],
    testIndices=[test_inds],
    net_type="resnet_101",
    augmenter_type="imgaug"
)
# %%
# train on A+B, shuffle 3
test_inds = []
train_inds = []
for i, path in enumerate(image_paths):
    if str(path[1]).endswith("_A"):
        train_inds.append(i)
    elif str(path[1]).endswith("_B"):
        train_inds.append(i)
    elif str(path[1]).endswith("_50_test"):
        test_inds.append(i)

deeplabcut.create_training_dataset(
    config_path,
    Shuffles=[3],
    trainIndices=[train_inds],
    testIndices=[test_inds],
    net_type="resnet_101",
    augmenter_type="imgaug"
)
# %%
# train on A+B+25, shuffle 4
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

deeplabcut.create_training_dataset(
    config_path,
    Shuffles=[4],
    trainIndices=[train_inds],
    testIndices=[test_inds],
    net_type="resnet_101",
    augmenter_type="imgaug"
)

# %%
help(deeplabcut.train_network)
# %%
