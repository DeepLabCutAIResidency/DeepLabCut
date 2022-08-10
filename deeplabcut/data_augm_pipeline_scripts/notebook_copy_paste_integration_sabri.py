# %%
import deeplabcut
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.datasets import Batch, PoseDatasetFactory
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
from deeplabcut.pose_estimation_tensorflow.datasets import augmentation
from deeplabcut.utils.auxfun_multianimal import extractindividualsandbodyparts
import os
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pickle
import imageio
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import random
from imgaug.augmentables import Keypoint, KeypointsOnImage

#from deeplabcut.data_augm_pipeline_scripts.copy_paste import CopyPaste
# empiezo a pasar algunas funciones para el copy paste

def crop_image(image,batch_joints, joint_ids):
    #input: image to data seg + joints ids + batch_joints (position of bdpts)
    #output: crop image + position ok kpts in the crop
    #self.main_cfg = read_config(os.path.join(self.cfg["project_path"], "config.yaml"))
    main_cfg = read_config(os.path.join('/media/data/trimice-dlc-2021-06-22_batchSize1/', "config.yaml"))
    animals, unique, multi = extractindividualsandbodyparts(main_cfg)
    
    random_animal = random.choice(animals)
    index = animals.index(random_animal)
    joint_random_animal = joint_ids[0][index]
    animals_not_crop = list(filter(lambda x: x != random_animal, animals))

    pos_bdpts_random_animal = batch_joints[0][index * len(joint_random_animal):index * len(joint_random_animal) +len(joint_random_animal) ]
    x = pos_bdpts_random_animal[:,0]
    y = pos_bdpts_random_animal[:,1]
    x1 = [x for x in x if str(x) != 'nan'] # remove nan
    y1 = [x for x in y if str(x) != 'nan']
    # plt.scatter(x1,y1)
    points = np.array([x1,y1])
    points = points.T
    #plt.scatter(points[:,0],points[:,1])
    Ymin = int(np.min(y1))
    Ymax = int(np.max(y1))
    Xmin = int(np.min(x1))
    Xmax = int(np.max(x1))
    crop_image = image[Ymin:Ymax,Xmin:Xmax]
    # bdpts from crop image
    x_new = [x - Xmin for x in x1]
    y_new = [ y - Ymin for y in y1]

    points_new = np.array([x_new,y_new])
    bdpts_crop = points_new.T
    return crop_image, bdpts_crop, animals_not_crop

def random_place(image,crop_image):
    #compute the new place of the crop image in the image
    x_top_left_crop = min(int(random.random()*image.shape[1]),
                        image.shape[1] - crop_image.shape[1])
    y_top_left_crop = min(int(random.random()*image.shape[0]),
                        image.shape[0] - crop_image.shape[0])
    polygon_crop = Polygon([(x_top_left_crop,y_top_left_crop), ( x_top_left_crop, y_top_left_crop+crop_image.shape[0]), 
                        (x_top_left_crop+crop_image.shape[1], y_top_left_crop+crop_image.shape[0]),
                        (x_top_left_crop+crop_image.shape[1],y_top_left_crop)])

    return x_top_left_crop, y_top_left_crop, polygon_crop

def occlued_overlap(animals_not_crop,batch_joints,joint_ids,animals,polygon_crop):
    dict_overlap ={}
    for i in animals_not_crop:   
        index = animals.index(i)
        joint_not_crop_animal = joint_ids[0][index]

        pos_bdpts_animal = batch_joints[0][index * len(joint_not_crop_animal):index * len(joint_not_crop_animal) +len(joint_not_crop_animal) ]

        x2 = pos_bdpts_animal[0::2]
        y2 = pos_bdpts_animal[1::2]
        x12 = [x for x in x2 if str(x) != 'nan'] # remove nan
        y12 = [x for x in y2 if str(x) != 'nan']
        # plt.scatter(x1,y1)

        points2 = np.array([x12,y12])
        points2 = points2
        for k in range(len(points2[0])):
            if polygon_crop.contains(Point(points2[0][k],points2[1][k])): #si overlap entonces nan
                points2[0][k] = np.nan
                points2[1][k] = np.nan

        dict_overlap[i] = points2


# %%
############################################################
config_path = '/media/data/trimice-dlc-2021-06-22_batchSize1/config.yaml' 
SHUFFLE_ID=1
TRAINING_SET_IDX=0
MODEL_PREFIX=''

train_cfg_path,\
_,_ = deeplabcut.return_train_network_path(config_path,
                                            shuffle=SHUFFLE_ID,
                                            trainingsetindex=TRAINING_SET_IDX, 
                                            modelprefix=MODEL_PREFIX)
train_cfg = load_config(str(train_cfg_path)) # cfg = load_config(config_yaml)

FLAG_PLOTTING = True

# %%
################################################
# Instantiate object from 'ImgaugPoseDataset' class (ok?)
# in multi-animal: MAImgaugPoseDataset!!!
# - 12 bodyparts, 3 animals
# - batch size = 8
dataset = PoseDatasetFactory.create(train_cfg)

# %%
###################################################
### Define pipeline --- we will define a custom pipeline here! 
# (in multi-animal property of MAImgaugPoseDataset class)            
# pipeline = dataset.build_augmentation_pipeline(height=target_size[0],  
#                                                 width=target_size[1], 
#                                                 apply_prob=0.5)

# to get pipeline defined for this dataset
# pipeline = dataset.pipeline # Sequential...

###########################################################
# Use an augmentation class?
# pipeline = iaa.Sequential(random_order=False)
# crop_sampling = train_cfg.get("crop_sampling", "hybrid")
# pipeline.add(augmentation.KeypointAwareCropToFixedSize(*dataset.default_size, 
#                                                        train_cfg.get("max_shift", 0.4), 
#                                                        crop_sampling))

# class CropToFixedSize(meta.Augmenter):

pipeline = iaa.Sequential(random_order=False)
#pipeline.add(CopyPaste())

# %%
########################################################
### Get batch (from multi-animal pose_imgaug, 'next_batch)
(batch_images, 
    joint_ids, 
    batch_joints, 
    inds_visible,
     data_items) = dataset.get_batch()

# %%
#### Plot original image if required
# If you would like to check the *original* images, script for saving
# the images with joints on:
if FLAG_PLOTTING:
    for i in range(dataset.batch_size):
        joints = batch_joints[i]
        kps = KeypointsOnImage([Keypoint(x=joint[0], y=joint[1]) for joint in joints],
                                shape=batch_images[i].shape)
        im = kps.draw_on_image(batch_images[i])
        # save original
        img_path = os.path.join(dataset.cfg["project_path"], str(i) + "_og.png")
        imageio.imwrite(img_path, im)
        print('Original image saved at: {}'.format(img_path))

# %%        
#########################################################################
#### Scale batch
## Scale is sampled only once (per batch) to transform all of the images into same size.
target_size, sm_size = dataset.calc_target_and_scoremap_sizes()
scale = np.mean(target_size/dataset.default_size)
augmentation.update_crop_size(dataset.pipeline, 
                              *target_size)

# if FLAG_PLOTTING:
#     for i in range(dataset.batch_size):
#         joints = batch_joints[i]
#         kps = KeypointsOnImage([Keypoint(x=joint[0], y=joint[1]) for joint in joints],
#                                 shape=batch_images[i].shape)
#         im = kps.draw_on_image(batch_images[i])
#         # save original
#         img_path = os.path.join(dataset.cfg["project_path"], str(i) + "_og_scale.png")
#         imageio.imwrite(img_path, im)
#         print('Original image saved at: {}'.format(img_path))                              

# %%
#########################################################################
### Transform batch
(batch_images, 
 batch_joints) = pipeline(images=batch_images, 
                          keypoints=batch_joints)

if FLAG_PLOTTING:
    for i in range(dataset.batch_size):
        joints = batch_joints[i]
        kps = KeypointsOnImage([Keypoint(x=joint[0], y=joint[1]) for joint in joints],
                                shape=batch_images[i].shape)
        im = kps.draw_on_image(batch_images[i])
        # save original
        img_path = os.path.join(dataset.cfg["project_path"], str(i) + "_transformed.png")
        imageio.imwrite(img_path, im)
        print('Transformed image saved at: {}'.format(img_path))     






# %%
#########################################################################
## Discard keypoints whose coordinates lie outside the cropped image ('joint_ids_valid')
batch_images = np.asarray(batch_images)
image_shape = batch_images.shape[1:3]
batch_joints_valid = []
joint_ids_valid = []
for joints, ids, visible in zip(batch_joints, joint_ids, inds_visible):
    joints = joints[visible]
    inside = np.logical_and.reduce(
        (
            joints[:, 0] < image_shape[1],
            joints[:, 0] > 0,
            joints[:, 1] < image_shape[0],
            joints[:, 1] > 0,
        )
    )
    batch_joints_valid.append(joints[inside])
    temp = []
    start = 0
    for array in ids:
        end = start + array.size
        temp.append(array[inside[start:end]])
        start = end
    joint_ids_valid.append(temp)

# %%
#########################################################################          
#### Plot if required
# If you would like to check the augmented images, script for saving
# the images with joints on:
if FLAG_PLOTTING:
    for i in range(dataset.batch_size):
        joints = batch_joints_valid[i]
        kps = KeypointsOnImage([Keypoint(x=joint[0], y=joint[1]) for joint in joints],
                                shape=batch_images[i].shape)
        im = kps.draw_on_image(batch_images[i])
        # save modified
        img_path = os.path.join(dataset.cfg["project_path"], str(i) + "_transformed2.png")
        imageio.imwrite(img_path, im)
        print('Transformed image saved at: {}'.format(img_path))  

# %%
#########################################################################
### Build dict of batch to return
batch = {Batch.inputs: batch_images.astype(np.float64)}
if dataset.has_gt:
    targetmaps = dataset.get_targetmaps_update(joint_ids_valid,
                                                batch_joints_valid,
                                                data_items,
                                                (sm_size[1], sm_size[0]),
                                                scale)
    batch.update(targetmaps)
# format as array?
batch = {key: np.asarray(data) for (key, data) in batch.items()}
batch[Batch.data_item] = data_items 


#########################################################################



