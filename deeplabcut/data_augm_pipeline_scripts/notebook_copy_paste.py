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
    # ATT! Vertices need to be in clockwise/anticlockwise order!
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

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
x1 = [x for x in x if str(x) != 'nan']
y1 = [x for x in y if str(x) != 'nan']
# plt.scatter(x1,y1)

points = np.array([x1,y1])
points = points.T
plt.scatter(points[:,0],points[:,1])

########################################################
### Compute convex hull 
hull = ConvexHull(points)
# hull_indices = np.unique(hull.simplices.flat)
hull_pts = points[hull.vertices,:]#points[hull_indices, :]
# for 2D, the hull vertices give the points in counter clockwise order


# plot convex hull
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

# plot vertices
plt.scatter(hull_pts[:,0], hull_pts[:,1], color='r',s=50)

# %%
# 
Y = int(np.min(y1))
H = int(np.max(y1))
X = int(np.min(x1))
W = int(np.max(x1))
cropped_image = image[Y:H,X:W]
plt.imshow(cropped_image)


# %%
import random
image2 = image
x_offset = random.randint(0, image2.shape[1])
y_offset = random.randint(0, image2.shape[0])

x_end = x_offset + cropped_image.shape[1]
y_end = y_offset + cropped_image.shape[0]

image2[y_offset:y_end,x_offset:x_end] = cropped_image
plt.imshow(image2)

# %%
# blend part

"""
image2 = image.copy()
img1 = cv2.resize(image,(800,600))
img2 = cv2.resize(cropped_image,(800,600))

blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
plt.imshow(blended)

"""