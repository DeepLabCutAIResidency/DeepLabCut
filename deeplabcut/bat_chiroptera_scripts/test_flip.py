# %%
import pandas as pd

image = '/media/data/jessy/MultiMouse-Daniel-2019-12-16/labeled-data/videocompressed0/img0038.png'
df = pd.read_hdf('/media/data/jessy/MultiMouse-Daniel-2019-12-16/labeled-data/videocompressed0/CollectedData_Daniel.h5')
kpts = df.columns.get_level_values('bodyparts').unique()
xy = df.loc[image.split('MultiMouse-Daniel-2019-12-16/')[1]]
xy = xy.to_numpy().reshape((-1, 12, 2))
xy_flat = xy.reshape(-1, 2)

# %%
import pandas as pd

image = '/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-08-03/labeled-data/IL5-534-Cam3_2020-06-25_000092Track1_A/97_96_92_c3_frame170.png'
df = pd.read_hdf('/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-08-03/labeled-data/IL5-534-Cam3_2020-06-25_000092Track1_A/CollectedData_DLC.h5')
kpts = df.columns.get_level_values('bodyparts').unique()
#xy = df.loc[image.split('geneva_protocol_paper_austin_2020_bat_data-DLC-2022-08-03/')[1]]
xy = df.iloc[16]
xy = xy.to_numpy().reshape((-1, 16, 2))
xy_flat = xy.reshape(-1, 2)
# %%
import imgaug.augmenters as iaa
from deeplabcut.pose_estimation_tensorflow.datasets import augmentation

pipeline = iaa.Sequential(random_order=False)
pipeline.add(augmentation.KeypointFliplr(kpts, [[0, 14]]))
# pipeline.add(iaa.Fliplr(1))
# %%
import matplotlib.pyplot as plt
from skimage import io

im = io.imread(image)
xy_ = xy.copy()
# %%
frame_, keypoints = pipeline(
    images=[im], keypoints=[xy_flat]
)
im = frame_[0]
xy_ = keypoints[0].reshape((-1, 16, 2))
# %%
fig, ax = plt.subplots()
ax.imshow(im)
ax.scatter(*xy_.T)
ax.scatter(*xy_[:, 14].T, c='darkorange')
#ax.annotate("lwt", xy_[0, 1])
#ax.annotate("leftear", xy_[1, 1])
#ax.annotate("leftear", xy_[2, 1])
ax.axis('off')
# %%
