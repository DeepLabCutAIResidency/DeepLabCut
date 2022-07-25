"""
Notebook that crops area around keypoints in a single-animal image and pastes it somewhere else in the same image

"""
# %%
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deeplabcut.utils import auxiliaryfunctions
import os
import cv2
import random

#############################################
def polygonArea(X, Y): 
    # X and Y are numpy arrays of size nrows=N, ncolumns=2

    n = X.shape[0]
    # Initialize area
    area = 0.0
    # Calculate value of shoelace formula
    j = n - 1
    for i in range(0,n):
        area += (X[j] + X[i]) * (Y[j] - Y[i])
        j = i   # j is previous vertex to i
    # Return absolute value
    return int(abs(area / 2.0))

# %% Load labelled data and plot an image
#############################################
config_path = '/media/data/stinkbugs-DLC-2022-07-15/config.yaml'
cfg = auxiliaryfunctions.read_config(config_path)
project_path = cfg["project_path"] # or: os.path.dirname(config_path) #dlc_models_path = os.path.join(project_path, "dlc-models")

# ideally: next bit from params and config?----
human_labels_filepath = '/media/data/stinkbugs-DLC-2022-07-15/data_augm_00_baseline/training-datasets/iteration-1/UnaugmentedDataSet_stinkbugsJul15/CollectedData_DLC.h5' #'/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/training-datasets/iteration-0/UnaugmentedDataSet_sabris-mouseJul6/CollectedData_nirel.h5'
df_human = pd.read_hdf(human_labels_filepath)
df_human = df_human.droplevel('scorer',axis=1) #df_human['DLC'][:].iloc[0,:]

## Plot a selected image
image_row_idx = 20
if type(df_human.index[image_row_idx]) is tuple:
    img_relative_path = os.path.join(*df_human.index[image_row_idx]) 
elif type(df_human.index[image_row_idx]) is str: 
    img_relative_path = df_human.index[image_row_idx] 
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
x1 = [x for x in x if str(x) != 'nan'] # remove nan
y1 = [x for x in y if str(x) != 'nan']


points = np.array([x1,y1])
points = points.T
plt.scatter(points[:,0],points[:,1])

########################################################
### Compute convex hull 
hull = ConvexHull(points)
hull_pts = points[hull.vertices,:]
# for 2D, the hull vertices give the points in counter clockwise order

# plot convex hull
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

# plot hull's vertices
plt.scatter(hull_pts[:,0], hull_pts[:,1], color='r',s=50)

#######################################
# Extract bounding box crop based on max min coords of keypoints
Ymin = int(np.min(y1))
Ymax = int(np.max(y1))
Xmin = int(np.min(x1))
Xmax = int(np.max(x1))
cropped_image = image[Ymin:Ymax,Xmin:Xmax]

# plt.imshow(cropped_image)

###########################
# Select crop region in original image and replace pixels with crop
image_w_replace = image

x_top_left_crop = min(int(random.random()*image.shape[1]),
                        image.shape[1] - cropped_image.shape[1]) # constrained so that crop stays within image!
y_top_left_crop = min(int(random.random()*image.shape[0]),
                        image.shape[0] - cropped_image.shape[0])


image_w_replace[y_top_left_crop:y_top_left_crop+cropped_image.shape[0],
                x_top_left_crop:x_top_left_crop+cropped_image.shape[1],
                0:] = cropped_image
plt.imshow(image_w_replace)

###########################################
# Blend crop with back
# blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
# plt.imshow(blended)


