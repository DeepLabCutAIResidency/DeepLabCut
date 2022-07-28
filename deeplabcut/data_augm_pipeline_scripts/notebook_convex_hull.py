# %%
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deeplabcut.utils import auxiliaryfunctions
import os
import cv2

# %%
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

# %%
########################################################
### Set config path for the project and path to human labels
config_path = '/media/data/stinkbugs-DLC-2022-07-15_COVERING/config.yaml' #'/media/data/Horses-Byron-2019-05-08/config.yaml'#'/media/data/stinkbugs-DLC-2022-07-15/config.yaml'
cfg = auxiliaryfunctions.read_config(config_path)
project_path = cfg["project_path"] # or: os.path.dirname(config_path) #dlc_models_path = os.path.join(project_path, "dlc-models")

# ideally: next bit from params and config?----
human_labels_filepath = '/media/data/stinkbugs-DLC-2022-07-15_COVERING/training-datasets/iteration-1/UnaugmentedDataSet_stinkbugsJul15/CollectedData_DLC.h5'#\
    # '/media/data/Horses-Byron-2019-05-08/training-datasets/iteration-0/UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5'
    # '/media/data/stinkbugs-DLC-2022-07-15/data_augm_00_baseline/training-datasets/iteration-1/UnaugmentedDataSet_stinkbugsJul15/CollectedData_DLC.h5' #'/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/training-datasets/iteration-0/UnaugmentedDataSet_sabris-mouseJul6/CollectedData_nirel.h5'
df_human = pd.read_hdf(human_labels_filepath)
df_human = df_human.droplevel('scorer',axis=1) #df_human['DLC'][:].iloc[0,:]


########################################################
### Plot a selected image
image_row_idx = 10 #250
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
plt.scatter(hull_pts[:,0],
            hull_pts[:,1],
            color='r',s=50)


######################################################
# plot scalebar
# area_chull = PolyArea(hull_pts[:,0],
#                       hull_pts[:,1])
# # scalebar_length = np.sqrt(area_chull)

scalebar_length = np.sqrt(polygonArea(hull_pts[:,0],
                                       hull_pts[:,1]))

print(scalebar_length)
# plot area as a square
# origin_square = np.mean(hull_pts,axis=0) -\
#                 np.array([0.5*scalebar_length, 0.5*scalebar_length])
# rectangle = plt.Rectangle(tuple(origin_square),
#                           scalebar_length2,
#                           scalebar_length2, 
#                           fc='blue',ec="blue",
#                           alpha=0.3)
# plt.gca().add_patch(rectangle)

# plot body scale estimate in x-axis
x_origin = 0
y_origin = image.shape[0] - 0.5
x_bar = [x_origin,  x_origin + scalebar_length]
y_bar = [y_origin]*2# repmat
plt.plot(x_bar,y_bar,
         color='r', linewidth=4)

# plot scale horiz
x_origin = np.mean(hull_pts,axis=0)[0]
y_origin = np.mean(hull_pts,axis=0)[1] - 0.5*scalebar_length
x_bar = [x_origin]*2
y_bar = [y_origin, y_origin + scalebar_length]

plt.plot(x_bar,y_bar,
         color='r', linewidth=4, linestyle=':')

# plot scale vert
x_origin = np.mean(hull_pts,axis=0)[0]  - 0.5*scalebar_length
y_origin = np.mean(hull_pts,axis=0)[1] 
x_bar = [x_origin, x_origin + scalebar_length]
y_bar = [y_origin]*2
plt.plot(x_bar,y_bar,
         color='r', linewidth=4, linestyle=':')

plt.show()

# %%
# # 
# Y = int(np.min(y1))
# H = int(np.max(y1))
# X = int(np.min(x1))
# W = int(np.max(x1))
# cropped_image = image[Y:H,X:W]
# print([X,Y,W,H])
# plt.imshow(cropped_image)
# # %%
# img1 = cv2.resize(image,(800,600))
# img2 = cv2.resize(cropped_image,(800,600))

# blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
# plt.imshow(blended)
# # %%
# from skimage.filters import gaussian

# def image_copy_paste(img, paste_img, alpha, blend=True, sigma=1):
#     if alpha is not None:
#         if blend:
#             alpha = gaussian(alpha, sigma=sigma, preserve_range=True)

#         img_dtype = img.dtype
#         alpha = alpha[..., None]
#         img = paste_img * alpha + img * (1 - alpha)
#         img = img.astype(img_dtype)

#     return img

# plt.imshow(image_copy_paste(cropped_image,cropped_image, alpha =1))
# # %%