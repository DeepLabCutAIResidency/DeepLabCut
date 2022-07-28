'''
From Assembler and  Assembly class at: DeepLabCut/deeplabcut/pose_estimation_tensorflow/lib/inferenceutils.py

'''
# %%
## Imports
import numpy as np
import warnings
import pandas as pd
from math import sqrt, erf

# from scipy.optimize import linear_sum_assignment
# from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist #, cdist
# from scipy.special import softmax
from scipy.stats import gaussian_kde, chi2

import deeplabcut
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils

# %%
#####################################
## Input data ?
train_data_file = \
    '/media/data/Horses-Byron-2019-05-08/training-datasets/iteration-0/UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5' # h5 file

# cfg # inference_cfg?

# select one row idx in labelled table,
#  after removing frames with less than 90% of kpts to compute probability
idx_sel_frame = 1000

# nan policy when computing mahalanobis distance
nan_policy_mahalanobis =  'little' # default is 'little'

# %%
##########################################
## Read labelled data into assembler
df = pd.read_hdf(train_data_file)
try:
    df.drop("single", level="individuals", axis=1, inplace=True)
except KeyError:
    pass

n_bpts = len(df.columns.get_level_values("bodyparts").unique())
if n_bpts == 1:
    warnings.warn("There is only one keypoint; skipping calibration...")
    # return

# %%
#########################################################
## Get matrix of bdprts  coords  per frame and select only frames in which almost all visible
# reshape joint data to matrix of (n_frames, n_bodyparts, 2) 
xy_all = df.to_numpy().reshape((-1, n_bpts, 2)) # (8114, 22, 2)
# Compute how complete each frame is (all kpts =1) 
frac_valid = np.mean(~np.isnan(xy_all), axis=(1, 2)) 
# Only keeps frames in which  more than 90% of the bodyparts are visible
xy = xy_all[frac_valid >= 0.9] # (4801, 22, 2)

# TODO Normalize dists by longest length?
# TODO Smarter imputation technique (Bayesian? Grassmann averages?)

# %%
#########################################################
## Compute pairwise distances between joints

dists = np.vstack([pdist(data, "sqeuclidean") for data in xy]) #(4801, 231) # for each frame, pass array of sorted keypoints # are these all in the same order? I guess so if data is 'sorted'
# replace missing data with mean
mu = np.nanmean(dists, axis=0) # mean bone length over all frames
missing = np.isnan(dists)
dists = np.where(missing, mu, dists)

# %%
#####################################################
## Compute PCA components here? (and then pass that to Gaussian KDE?)
# OJO normalise before PCA


# %%
#####################################################
## Estimate pdf using kernel density estimation
# Kernel density estimation is a way to estimate the probability density function (PDF) of a random variable 
# in a non-parametric way. gaussian_kde works for both uni-variate and multi-variate data. It includes automatic 
# bandwidth determination. The estimation works best for a unimodal distribution; bimodal or multi-modal distributions 
# tend to be oversmoothed.
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html 
try:
    kde = gaussian_kde(dists.T) # if a 2D array input should be # dimensions, # data
    kde.mean = mu
    # self._kde = kde
    # self.safe_edge = True
except np.linalg.LinAlgError:
    # Covariance matrix estimation fails due to numerical singularities
    warnings.warn("The assembler could not be robustly calibrated. Continuing without it...")

# - kde.dataset = input data
# - kde.d = dimensions of the space (variables)
# - kde.n = number of datapoints
# - kde.neff =  effective number of datapoints
# - kde.factor = bandwidth factor
# - kde._data_covariance = covariance matrix of the input data
# - kde.covariance = The covariance matrix of dataset, scaled by the calculated bandwidth
# %%
#########################################
# Calculate Mahalanobis distance to distribution ---does it take nans?
# (method of Assembler)

# Cast columns of dtype 'object' to float to avoid TypeError
# further down in _parse_ground_truth_data.
cols = df.select_dtypes(include="object").columns
if cols.to_list():
    df[cols] = df[cols].astype("float")
# n_individuals = 1 #len(df.columns.get_level_values("individuals").unique())
n_bodyparts = len(df.columns.get_level_values("bodyparts").unique())
data = df.to_numpy().reshape((df.shape[0], n_bodyparts, -1)) #df.to_numpy().reshape((df.shape[0], n_individuals, n_bodyparts, -1))

# instantiate Assembly object from keypoint data for one frame
row = xy[idx_sel_frame] #xy_all[idx_sel_frame]
assembly = inferenceutils.Assembly.from_array(row) # row=n_bpts, n_cols  # OJO not sure what row is

# %%
# compute pairwise distances in selected skeleton and the deviation of the vector wrt to the mean? 
# It is a multi-dimensional generalization of the idea of measuring how many standard deviations away P is from the mean of D.
#  https://en.wikipedia.org/wiki/Mahalanobis_distance 
dists = assembly.calc_pairwise_distances() - kde.mean #self._kde.mean
mask = np.isnan(dists) # slc nans

# Distance is undefined if the assembly is empty
if not len(assembly) or mask.all():
    mahal_dist = np.inf
    proba = 0

# Deal with nans
if nan_policy_mahalanobis == "little":
    inds = np.flatnonzero(~mask) # Return indices that are non-zero in the flattened version of a.
    dists = dists[inds] #keep only those that are not nan
    inv_cov = kde.inv_cov[np.ix_(inds, inds)] # self._kde.inv_cov[np.ix_(inds, inds)] # Using ix_ one can quickly construct index arrays that will index the cross product.
    # Correct distance to account for missing observations
    factor = kde.d / len(inds)
else:
    # Alternatively, reduce contribution of missing values to the Mahalanobis
    # distance to zero by substituting the corresponding means.
    dists[mask] = 0 # set distance of nans to 0
    mask.fill(False) # sets all the mask to false? (as if no nans)
    inv_cov = kde.inv_cov
    factor = 1

# compute Mahalanobis distance and prob
dot = dists @ inv_cov # conventional matrix multiplication

mahal = factor * sqrt(np.sum((dot * dists), axis=-1))
proba = 1 - chi2.cdf(mahal, np.sum(~mask))
# %%
