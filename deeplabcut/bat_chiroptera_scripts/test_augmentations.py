""""
Adapted from code written by Sabrina
"""

# %%
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import glob
import matplotlib.pyplot as plt
import cv2
images = []
images_path = glob.glob("/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-19/labeled-data/IL5-519-Cam2_2020-06-25_000007_cropped/*.png")
for img_path in images_path:
    img = cv2.imread(img_path)
    images.append(img)

motion_blur_param = {"k": 7, "angle": (-90, 90)}

augmentation = [
    #iaa.GaussianBlur(sigma=(0,0)),
    iaa.MultiplyAndAddToBrightness(mul=(0.8, 0.8), add=(-20,-20))
    #iaa.Rotate(rotate=(-180,-180))
    #iaa.CoarseDropout(0.02, size_percent=0.3, per_channel=0.5),
    #iaa.ElasticTransformation(sigma=5),
    #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
    #iaa.Grayscale(alpha=(0.5, 1.0)),
    #iaa.FastSnowyLandscape(lightness_threshold=140,lightness_multiplier=2.5 ),
    #iaa.SnowflakesLayer(density=(0.005, 0.075),
    #            density_uniformity=(0.3, 0.9),
    #            flake_size=(0.2, 0.7), flake_size_uniformity=(0.4, 0.8),
    #            angle=(-30, 30), speed=(0.007, 0.03), blur_sigma_fraction=(0.0001, 0.001)),
    #iaa.Fog(),
    #iaa.CloudLayer(intensity_mean=(196, 255),
    #            intensity_freq_exponent=(-2.5, -2.0),
    #            intensity_coarse_scale=10,
    #            alpha_min=0, #this parameter tiene un cambio grande
    #            alpha_multiplier=(0.25, 0.75),
    #            alpha_size_px_max=(2, 8),
    #            alpha_freq_exponent=(-2.5, -2.0),
    #            sparsity=(0.8, 1.0),
    #            density_multiplier=(0.5, 1.0),),
    #iaa.RainLayer(density=(0.03, 0.14),
    #        density_uniformity=(0.8, 1.0),
    #        drop_size=(0.01, 0.02),
    #        drop_size_uniformity=(0.2, 0.5),
    #        angle=(-15, 15),
    #        speed=(0.04, 0.20),
    #        blur_sigma_fraction=(0.001, 0.001)),
    #iaa.MotionBlur(**motion_blur_param)
  ]

# %%
for k in augmentation:
    aug = iaa.Sequential(k)
    augmented_images = aug(images=images)
    for img in augmented_images:
        fig = plt.figure(figsize =(10,12))
        plt.imshow(img[:,:,::-1])
        plt.show()
# %%
