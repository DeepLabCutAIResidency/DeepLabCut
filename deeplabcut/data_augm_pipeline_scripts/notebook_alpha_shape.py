# %%
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deeplabcut.utils import auxiliaryfunctions
import os
import cv2

import os
import sys
import pandas as pd
import numpy as np
from descartes import PolygonPatch
import matplotlib.pyplot as plt

def PolyArea(x,y): 
    # https://en.wikipedia.org/wiki/Shoelace_formula
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

# %% Read groundtruth data
config_path = '/media/data/stinkbugs-DLC-2022-07-15/config.yaml'
cfg = auxiliaryfunctions.read_config(config_path)
project_path = cfg["project_path"] # or: os.path.dirname(config_path) #dlc_models_path = os.path.join(project_path, "dlc-models")

# ideally: next bit from params and config?----
human_labels_filepath ='/media/data/stinkbugs-DLC-2022-07-15/data_augm_00_baseline/training-datasets/iteration-1/UnaugmentedDataSet_stinkbugsJul15/CollectedData_DLC.h5' #'/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/training-datasets/iteration-0/UnaugmentedDataSet_sabris-mouseJul6/CollectedData_nirel.h5'
df_human = pd.read_hdf(human_labels_filepath)
df_human = df_human.droplevel('scorer',axis=1) #df_human['DLC'][:].iloc[0,:]

image_row_idx = 50
img_relative_path = os.path.join(*df_human.index[image_row_idx])
labeled_data_path = os.path.join(project_path, img_relative_path)

fig, ax = plt.subplots(figsize=(10,10))
image = cv2.imread(labeled_data_path)


ax.imshow(image)


lts = list(df_human.iloc[image_row_idx,:])
x = lts[0::2]
y = lts[1::2]
x1 = [x for x in x if str(x) != 'nan']
y1 = [x for x in y if str(x) != 'nan']
plt.scatter(x1,y1)
points_2d = np.array([x1,y1])
points_2d = points_2d.T

import alphashape



alpha_shape = alphashape.alphashape(points_2d, 0.021)
#alpha_shape



sys.path.insert(0, os.path.dirname(os.getcwd()))
import alphashape
ax.scatter(*zip(*points_2d))
ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))

# %%
# %%