#/home/sofia/miniconda/envs/deeplabcut-res/lib/python3.8/site-packages/imgaug/augmenters/size.py
import imgaug as ia
import imgaug.augmenters as iaa  #from imgaug.augmenters import Augmenter
import numpy as np
import random
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
from deeplabcut.pose_estimation_tensorflow.datasets import augmentation
from deeplabcut.utils.auxfun_multianimal import extractindividualsandbodyparts

def crop_image(main_cfg,
               image,
               batch_joints, 
               joint_ids):
    #input: image to data seg + joints ids + batch_joints (position of bdpts)
    #output: crop image + position ok kpts in the crop
    #self.main_cfg = read_config(os.path.join(self.cfg["project_path"], "config.yaml"))
    # main_cfg = read_config(os.path.join('/media/data/trimice-dlc-2021-06-22_batchSize1/', "config.yaml"))
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

# def random_place(image,crop_image):
#     #compute the new place of the crop image in the image
#     x_top_left_crop = min(int(random.random()*image.shape[1]),
#                         image.shape[1] - crop_image.shape[1])
#     y_top_left_crop = min(int(random.random()*image.shape[0]),
#                         image.shape[0] - crop_image.shape[0])
#     polygon_crop = Polygon([(x_top_left_crop,y_top_left_crop), ( x_top_left_crop, y_top_left_crop+crop_image.shape[0]), 
#                         (x_top_left_crop+crop_image.shape[1], y_top_left_crop+crop_image.shape[0]),
#                         (x_top_left_crop+crop_image.shape[1],y_top_left_crop)])

#     return x_top_left_crop, y_top_left_crop, polygon_crop

# def occlued_overlap(animals_not_crop,batch_joints,joint_ids,animals,polygon_crop):
#     dict_overlap ={}
#     for i in animals_not_crop:   
#         index = animals.index(i)
#         joint_not_crop_animal = joint_ids[0][index]

#         pos_bdpts_animal = batch_joints[0][index * len(joint_not_crop_animal):index * len(joint_not_crop_animal) +len(joint_not_crop_animal) ]

#         x2 = pos_bdpts_animal[0::2]
#         y2 = pos_bdpts_animal[1::2]
#         x12 = [x for x in x2 if str(x) != 'nan'] # remove nan
#         y12 = [x for x in y2 if str(x) != 'nan']
#         # plt.scatter(x1,y1)

#         points2 = np.array([x12,y12])
#         points2 = points2
#         for k in range(len(points2[0])):
#             if polygon_crop.contains(Point(points2[0][k],points2[1][k])): #si overlap entonces nan
#                 points2[0][k] = np.nan
#                 points2[1][k] = np.nan

#         dict_overlap[i] = points2

######################################################
class CopyPaste(iaa.meta.Augmenter): #-----https://imgaug.readthedocs.io/en/latest/source/api_augmenters_blend.html#imgaug.augmenters.blend.BlendAlphaMask
    ############################################################################
    def __init__(self, 
                 config,#----n animals
                 p=1,
                 seed=None, 
                 name=None,
                 random_state="deprecated", 
                 deterministic="deprecated"):
        # parent class
        super(CopyPaste, self).__init__(seed=seed, 
                                        name=name,
                                        random_state=random_state, 
                                        deterministic=deterministic)
        # attributes
        self.p = ia.parameters.handle_probability_param(p, "p")
        self.config = config
    #############################################################################
    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        ###########################################
        # - batch: ['images', 'keypoints',....]
        # - len(batch.keypoints[0]) = 36 -----> for every image: 36 keypoints. TO CHECK: sorted? nan if occluded?
        samples = self.p.draw_samples((batch.nb_rows,),
                                      random_state=random_state)
        for i, sample in enumerate(samples):
            if sample >= 0.5:
                if batch.images is not None: # batch.images: list of np arrays
                    # get one sample from batch
                    image = batch.images[0]
                    # numpy split array or reshape! keypoints!

                    # /home/sofia/DeepLabCut/deeplabcut/pose_estimation_tensorflow/lib/trackingutils.py -- bbox computation from mpts
                    from deeplabcut.pose_estimation_tensorflow.lib.trackingutils import calc_bboxes_from_keypoints

                    ####################
                    # crop from that image
                    # (cropped_image, 
                    # bdpts_crop, 
                    # animals_not_crop) = crop_image(self.config,
                    #                                 image,
                    #                                 batch_joints, 
                    #                                 joint_ids)
                    
                    # paste on that image 


                    # We currently do not use flip.flipud() here, because that
                    # saves a function call.
                    # batch.images[i] = batch.images[i][::-1, ...]

                if batch.heatmaps is not None:
                    batch.heatmaps[i].arr_0to1 = \
                        batch.heatmaps[i].arr_0to1[::-1, ...]

                if batch.segmentation_maps is not None:
                    batch.segmentation_maps[i].arr = \
                        batch.segmentation_maps[i].arr[::-1, ...]

                if batch.keypoints is not None:
                    kpsoi = batch.keypoints[i]
                    height = kpsoi.shape[0]
                    for kp in kpsoi.keypoints:
                        kp.y = height - float(kp.y)

                if batch.bounding_boxes is not None:
                    bbsoi = batch.bounding_boxes[i]
                    height = bbsoi.shape[0]
                    for bb in bbsoi.bounding_boxes:
                        # after flip, y1 ends up right of y2
                        y1, y2 = bb.y1, bb.y2
                        bb.y1 = height - y2
                        bb.y2 = height - y1

                if batch.polygons is not None:
                    psoi = batch.polygons[i]
                    height = psoi.shape[0]
                    for poly in psoi.polygons:
                        # TODO maybe reverse the order of points afterwards?
                        #      the flip probably inverts them
                        poly.exterior[:, 1] = height - poly.exterior[:, 1]

                if batch.line_strings is not None:
                    lsoi = batch.line_strings[i]
                    height = lsoi.shape[0]
                    for ls in lsoi.line_strings:
                        # TODO maybe reverse the order of points afterwards?
                        #      the flip probably inverts them
                        ls.coords[:, 1] = height - ls.coords[:, 1]

        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p]
