'''
From Assembler and  Assembly class at: DeepLabCut/deeplabcut/pose_estimation_tensorflow/lib/inferenceutils.py

'''
# %%
## Imports
import numpy as np
import warnings
import pandas as pd
from math import sqrt, erf
import os
import random
import matplotlib.pyplot as plt


# from scipy.optimize import linear_sum_assignment
# from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, mahalanobis #, cdist
# from scipy.special import softmax
from scipy.stats import gaussian_kde, chi2


import deeplabcut
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils

# %%
########################################################
## Input data ?
train_data_file = \
    '/home/sofia/datasets/Horses-Byron-2019-05-08/training-datasets/iteration-0/UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5' # h5 file

# cfg # inference_cfg?

list_horseIDs_for_kde = [*range(15)]
# select one row idx in labelled table,
#  after removing frames with less than 90% of kpts to compute probability
# idx_sel_frame = 1000

# nan policy when computing mahalanobis distance
# nan_policy_mahalanobis =  'little' # default is 'little'

# mahalanobis
chi2_percentile = 0.9

# seed:
random.seed(3)

# %%
######################################################
## Read labelled data 
df_all_horses = pd.read_hdf(train_data_file)

# drop indiv-level if multi-animal
try:
    df_all_horses.drop("single", level="individuals", axis=1, inplace=True) # for multi-animal
except KeyError:
    pass

# get n of bodyparts per animal
n_bpts = len(df_all_horses.columns.get_level_values("bodyparts").unique())
if n_bpts == 1:
    warnings.warn("There is only one keypoint; skipping calibration...")
    # return

# add Horse ID as index
set_horses_ID_str = np.unique([v for (u,v,w) in df_all_horses.index])
dict_horse_ID_str_to_int = {el:j for j,el in enumerate(set_horses_ID_str)}
# keep frame path info
df_all_horses['framePath']=[os.path.join(*el) for el in df_all_horses.index] 
df_all_horses.insert(0,'horseID',
                    [dict_horse_ID_str_to_int[v] for (u,v,w) in df_all_horses.index],
                    allow_duplicates=True)
df_all_horses.set_index('framePath', inplace=True)

# %%
# L=[v for (u,v,w) in df_all_horses.index]
# for i, el in enumerate(set_horses_ID_str):
#     if i == len(set_horses_ID_str)-1:
#         print('end')
#         break
#     elif L.index(set_horses_ID_str[i+1])  != L.index(el) + L.count(el):
#         break
#     else:
#         continue



# %%
#########################################################################
# Select one horse only and get matrix of bodyparts coords per frame
# to select rows from a specific Horse: df.loc[df['Horse ID']==0]
# https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values
df_slc_horses = df_all_horses.loc[df_all_horses['horseID'].isin(list_horseIDs_for_kde)]
df_other_horses = df_all_horses.loc[~df_all_horses['horseID'].isin(list_horseIDs_for_kde)]

## Get matrix of bdprts  coords  per frame 
# Reshape joint data to matrix of (n_frames, n_bodyparts, 2) 
bdpts_per_frame_XY_slc_horses = df_slc_horses.drop('horseID',axis=1,level=0).to_numpy().reshape((-1, n_bpts, 2)) # (8114, 22, 2)
bdpts_per_frame_XY_other_horses = df_other_horses.drop('horseID',axis=1,level=0).to_numpy().reshape((-1, n_bpts, 2)) 

## Add fraction of kpts visible per frame
# Compute how complete each frame is (all kpts =1) 
for bdpts_arr,df in zip([bdpts_per_frame_XY_slc_horses, bdpts_per_frame_XY_other_horses],
                        [df_slc_horses, df_other_horses]):
    frac_of_vis_kpts_per_frame = np.mean(~np.isnan(bdpts_arr), axis=(1, 2)) 

    # add to dataframe
    df['fraction_vis_kpts'] = frac_of_vis_kpts_per_frame

# %%
#################################################################################
## Select only frames in which over 90% of kpts are visible
bdpts_per_frame_XY_slc_horses_valid = bdpts_per_frame_XY_slc_horses[df_slc_horses['fraction_vis_kpts'] >= 0.9] # (4801, 22, 2)


# inspecting how many frames discarded...
print(sum(df_slc_horses['fraction_vis_kpts']  >= 0.9)/len(bdpts_per_frame_XY_slc_horses))

plt.hist(df_slc_horses['fraction_vis_kpts'] , 
         density=False,
         bins = np.arange(0.0,1.1,0.1))
plt.xlabel('fraction of visible keypoints')
plt.xticks(np.arange(0.0,1.1,0.1))
plt.ylabel('n frames (N = {})'.format(len(bdpts_per_frame_XY_slc_horses)))
plt.vlines(np.median(df_slc_horses['fraction_vis_kpts'] ), 
          0, 0.65*len(bdpts_per_frame_XY_slc_horses),
          color='r',
          linestyle='--',
          label='median')
# plt.grid()
# plt.ylim([0,100])
plt.legend(loc='upper left')
# plt.title('Horse ID: {}'.format(horseID_for_kde))
plt.show()


plt.plot(df_slc_horses['fraction_vis_kpts'] ,
        '.-')
plt.hlines(0.9,
           0,len(df_slc_horses['fraction_vis_kpts'] ),
           color='r',
           linestyle=':',
           label='threshold for kde')
plt.xlabel('frame idx')
plt.xticks(np.arange(0,len(df_slc_horses),500), np.arange(0,len(df_slc_horses),500))
plt.ylabel('fraction of visible keypoints')
# plt.title('Horse ID: {} ({})'.format(horseID_for_kde, 
#                                      [el.split('/')[1] for el in df_slc_horses.index][0]))
plt.show()
# TODO Normalize dists by longest length?
# TODO Smarter imputation technique (Bayesian? Grassmann averages?)

# %%
#########################################################
## Compute pairwise distances between kpts -limbs lengths, replacing missing data with mean limb length
# (n selected frames, nchoosek(22,2) )
pairwise_sq_dists_per_frame= np.vstack([pdist(data, "sqeuclidean") \
                                     for data in bdpts_per_frame_XY_slc_horses_valid]) #(4801, 231) # for each frame, pass array of sorted keypoints # are these all in the same order? I guess so if data is 'sorted'
# replace missing data with mean
mu = np.nanmean(pairwise_sq_dists_per_frame, axis=0) # mean limb length over all frames
missing = np.isnan(pairwise_sq_dists_per_frame)
pairwise_sq_dists_per_frame_no_nans = np.where(missing, mu, pairwise_sq_dists_per_frame)

plt.matshow(pairwise_sq_dists_per_frame_no_nans)
plt.legend()
plt.xlabel('sorted pairs of kpts')
plt.ylabel('selected frames')
plt.colorbar()
# plt.title('pairwise square distances (px**2)')
plt.show()
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

kde_slc_horses = gaussian_kde(pairwise_sq_dists_per_frame_no_nans.T) # if a 2D array input should be # dimensions, # data
kde_slc_horses.mean = mu

# try:
    # kde_slc_horses = gaussian_kde(pairwise_sq_dists_per_frame_no_nans.T) # if a 2D array input should be # dimensions, # data
    # kde_slc_horses.mean = mu
    # self._kde = kde
    # self.safe_edge = True
# except np.linalg.LinAlgError:
#     # Covariance matrix estimation fails due to numerical singularities
#     warnings.warn("The assembler could not be robustly calibrated. Continuing without it...")

# - kde.dataset = input data
# - kde.d = dimensions of the space (variables)
# - kde.n = number of datapoints
# - kde.neff =  effective number of datapoints
# - kde.factor = bandwidth factor
# - kde._data_covariance = covariance matrix of the input data
# - kde.covariance = The covariance matrix of dataset, scaled by the calculated bandwidth
# %%
##################################################################################
### Select one frame from the set of remaining horses
# chose from frames in which all kpts are fully viz
df_other_horses_fully_viz = df_other_horses.loc[df_other_horses['fraction_vis_kpts']==1] 
bdpts_sel_frames_XY_other_horses = df_other_horses_fully_viz.drop(['horseID','fraction_vis_kpts'],axis=1,level=0).to_numpy().reshape(-1,n_bpts,2)
# bdpts_one_frame_XY_other_horses = bdpts_one_frame_XY_other_horses[0,:,:] #random.randint(0,len(df_other_horses_fully_viz))]

# compute pairwise sq-distances
# (n frames)
pairwise_sq_dists_one_frame = np.asarray([pdist(arr, metric="sqeuclidean") 
                                         for arr in bdpts_sel_frames_XY_other_horses])

# plot frame with keypoints
#

# Evaluate
# ---evaluate would be: how likely is this distribution to observe this precise pose?
# # instead we ask: how many std away from the mean this point is
# prob_from_eval = kde_slc_horses.evaluate(pairwise_sq_dists_one_frame.T) #(# of dimensions, # of points)-array

# plt.plot(prob_from_eval) 
# plt.show()

# # Evaluate on same---zero???
# prob_from_eval_inner = kde_slc_horses.evaluate(pairwise_sq_dists_per_frame_no_nans.T) #(# of dimensions, # of points)-array
# plt.plot(prob_from_eval_inner) 
# plt.show()

# %%
################################
# Compute Mahalanobis distance to mean of kde distribution
# why not the same as Jessy's method? :?

d_mahal = np.zeros((pairwise_sq_dists_one_frame.shape[0],1))
for i,d in enumerate(d_mahal):
    d_mahal[i,:] = mahalanobis(pairwise_sq_dists_one_frame[i,:],
                                            kde_slc_horses.mean,
                                            kde_slc_horses._data_inv_cov) #inv_cov # kde_slc_horses._data_inv_cov


plt.plot(d_mahal)
plt.ylim([0,100])
plt.xlabel('frame ID')
plt.ylabel('Mahalanobis distance d (px)')

# add percent point fn (percentiles, inverse of cdf: ppf(q, df, loc=0, scale=1))
# d2 follows X**2 distrib with dof = n of dimensions
sq_d_mahal_percentile = chi2.ppf(chi2_percentile, 
                                 pairwise_sq_dists_one_frame.shape[1]) # loc=0? scale=1? stats.ncx2?

plt.hlines(sqrt(sq_d_mahal_percentile),
           0,len(d_mahal),
           'r')

# %%
###################################################
## Compute prob
# prob of the random variable d**2 being above the observed value?
# d is Mahalanobis distance, d**2 follows X**2 distribution
proba = np.zeros(d_mahal.shape)
for i,d in enumerate(proba):
    
    # COMPUTE p1=1-CDF.CHISQ(d-squared,nobsvar).
    # https://stats.stackexchange.com/questions/28593/mahalanobis-distance-distribution-of-multivariate-normally-distributed-points
    proba[i,:] = 1 - chi2.cdf(d_mahal[i,:]**2, 
                              pairwise_sq_dists_one_frame.shape[1]) 
   

plt.plot(proba,
         '.-')
# plt.ylim([0,100])
plt.xlabel('frame ID')
plt.ylabel('prob of d**2 being above the observed value')

plt.hlines(chi2_percentile,
           0,len(proba),
           'r')
# %%
#####################################################################################
# Calculate Mahalanobis distance between selected pose and reference distribution 
# (method of Assembler)
# compute pairwise distances in selected skeleton and the deviation of the vector wrt to the mean? 
# It is a multi-dimensional generalization of the idea of measuring how many standard deviations away P is from the mean of D.
#  https://en.wikipedia.org/wiki/Mahalanobis_distance 
dists = pairwise_sq_dists_one_frame - kde_slc_horses.mean #assembly.calc_pairwise_distances() - kde_slc_horses.mean #self._kde.mean
mask = np.isnan(dists) # slc nans

# Deal with nans
if nan_policy_mahalanobis == "little":
    inds = np.flatnonzero(~mask) # Return indices that are non-zero in the flattened version of a.
    dists = dists[inds] #keep only those that are not nan
    inv_cov = kde_slc_horses.inv_cov[np.ix_(inds, inds)] # self._kde.inv_cov[np.ix_(inds, inds)] # Using ix_ one can quickly construct index arrays that will index the cross product.
    # Correct distance to account for missing observations
    factor = kde_slc_horses.d / len(inds)
else:
    # Alternatively, reduce contribution of missing values to the Mahalanobis
    # distance to zero by substituting the corresponding means.
    dists[mask] = 0 # set distance of nans to 0
    mask.fill(False) # sets all the mask to false? (as if no nans)
    inv_cov = kde_slc_horses.inv_cov
    factor = 1

# compute Mahalanobis distance and prob
dot = dists @ inv_cov # conventional matrix multiplication
mahal = factor * sqrt(np.sum(((dists @ inv_cov) * dists), axis=-1)) # factor * sqrt(np.sum((dot * dists), axis=-1))
proba = 1 - chi2.cdf(mahal, np.sum(~mask)) #---- i think it should be mahal**2?
print(proba)

# compare to
# prob_eval = kde_slc_horses.evaluate()?

# %%
