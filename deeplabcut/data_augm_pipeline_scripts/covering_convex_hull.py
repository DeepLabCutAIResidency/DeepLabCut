# %%
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import imgaug.augmenters as iaa
import glob
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
# %%
config_path = '/media/data/stinkbugs-DLC-2022-07-15/config.yaml'
cfg = auxiliaryfunctions.read_config(config_path)
project_path = cfg["project_path"] # or: os.path.dirname(config_path) #dlc_models_path = os.path.join(project_path, "dlc-models")

# ideally: next bit from params and config?----
human_labels_filepath = '/media/data/stinkbugs-DLC-2022-07-15/data_augm_00_baseline/training-datasets/iteration-1/UnaugmentedDataSet_stinkbugsJul15/CollectedData_DLC.h5' #'/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/training-datasets/iteration-0/UnaugmentedDataSet_sabris-mouseJul6/CollectedData_nirel.h5'
df_human = pd.read_hdf(human_labels_filepath)
df_human = df_human.droplevel('scorer',axis=1) #df_human['DLC'][:].iloc[0,:]

## Plot a selected image
image_row_idx = 30
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
area_bug = polygonArea(hull_pts[:,0], hull_pts[:,1])
scalebar_length = np.sqrt(polygonArea(hull_pts[:,0], hull_pts[:,1]))
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
# covering
augmentation = iaa.Sequential(([iaa.CoarseDropout(
                           ),
                       ]))

augmented_images = augmentation(images=image)
plt.imshow(augmented_images[:,:,::-1])
plt.show()
# %%
import imgaug as ia
images = np.array(
    [ia.quokka(size=(64, 64)) for _ in range(32)],
    dtype=np.uint8
)

# %%
image_as_batch=np.expand_dims(image,axis=0)
lst= np.arange(0,1,0.05)
#for i in lst:
augmentation2 =iaa.Sequential(iaa.Cutout(cval =0,size= 1*np.sqrt(area_bug)/864,
            squared=True))
#augmentation.augment_image(image)
img_aug = augmentation2.augment_image(image_as_batch[0,:,:])
plt.imshow(img_aug)
plt.show()

# %%
cov = np.linspace(0., 2 ,num = 5)
for i in cov:
    pipeline =iaa.Sequential(random_order=False)
    pipeline.add(iaa.CoarseDropout(0.1,size_percent=1/(i*np.sqrt(area_bug))))
    #augmentation.augment_image(image)
    imgs_augm = pipeline(images=image_as_batch)
    plt.imshow(imgs_augm[0,:,:])
    plt.show()



# %%
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


ia.seed(1)

# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.
images = np.array(
    [ia.quokka(size=(64, 64)) for _ in range(32)],
    dtype=np.uint8
)

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order

images_aug = seq(images=images)
plt.imshow(images_aug[0,:,:,:])
# %%
