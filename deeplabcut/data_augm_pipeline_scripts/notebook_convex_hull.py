# %%
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deeplabcut.utils import auxiliaryfunctions
import os
import cv2


def PolyArea(x,y): 
    # https://en.wikipedia.org/wiki/Shoelace_formula
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


########################################################
### Set config path for the project and path to human labels
config_path = '/media/data/stinkbugs-DLC-2022-07-15/config.yaml'
cfg = auxiliaryfunctions.read_config(config_path)
project_path = cfg["project_path"] # or: os.path.dirname(config_path) #dlc_models_path = os.path.join(project_path, "dlc-models")

# ideally: next bit from params and config?----
human_labels_filepath ='/media/data/stinkbugs-DLC-2022-07-15/data_augm_00_baseline/training-datasets/iteration-1/UnaugmentedDataSet_stinkbugsJul15/CollectedData_DLC.h5' #'/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/training-datasets/iteration-0/UnaugmentedDataSet_sabris-mouseJul6/CollectedData_nirel.h5'
df_human = pd.read_hdf(human_labels_filepath)
df_human = df_human.droplevel('scorer',axis=1) #df_human['DLC'][:].iloc[0,:]

########################################################
### Plot a selected image
image_row_idx = 50
img_relative_path = os.path.join(*df_human.index[image_row_idx])
labeled_data_path = os.path.join(project_path, img_relative_path)

image = cv2.imread(labeled_data_path)
fig = plt.figure(figsize=(10,10))
plt.imshow(image)

########################################################
### Get keypoints for this image
# df_human_wo_nan = df_human.dropna(axis=0).reset_index(drop=True) ----review! why it didnt work?
lts = list(df_human.iloc[image_row_idx,:])
x = lts[0::2]
y = lts[1::2]
x1 = [x for x in x if str(x) != 'nan']
y1 = [x for x in y if str(x) != 'nan']
# plt.scatter(x1,y1)

points = np.array([x1,y1])
points = points.T
plt.scatter(points[:,0],points[:,1])

########################################################
### Compute convex hull 
hull = ConvexHull(points)
hull_indices = np.unique(hull.simplices.flat)
hull_pts = points[hull_indices, :]

# plot convex hull
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

# plot vertices
plt.scatter(hull_pts[:,0],
            hull_pts[:,1],
            color='r',s=50)

########################################################
# plot scalebar
area_chull = PolyArea(hull_pts[:,0],
                      hull_pts[:,1])
scalebar_length = np.sqrt(area_chull)

x_origin = 50
y_origin = 50
x_bar = [x_origin, x_origin + scalebar_length]# repmat
y_bar = [y_origin]*2
plt.plot(x_bar,y_bar,
         color='r', linewidth=4)
# %%
