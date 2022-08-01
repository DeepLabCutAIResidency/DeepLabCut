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
from scipy.stats import gaussian_kde, chi2, ncx2


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

# sq_dist_diag_in_px2 = 288**2 + 162**2
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

# add Horse ID as column (level 0 ok?)
set_horses_ID_str = np.unique([v for (u,v,w) in df_all_horses.index])
dict_horse_ID_str_to_int = {el:j for j,el in enumerate(set_horses_ID_str)}
df_all_horses.insert(0,'horseID', # ('','horseID',''),
                    [dict_horse_ID_str_to_int[v] for (u,v,w) in df_all_horses.index],
                    allow_duplicates=True,)

# keep frame path info as index
df_all_horses.loc[:,'framePath']=[os.path.join(*el) for el in df_all_horses.index] 
df_all_horses.set_index('framePath', inplace=True)

# df_all_horses['horseID'] = [dict_horse_ID_str_to_int[v] for (u,v,w) in df_all_horses.index]
# df_all_horses.set_index('horseID', append=True, inplace=True)
# df_all_horses.reorder_levels(['horseID'])



# %%
#########################################################################
# Add fraction of kpts visible per frame

## Split selected and other horses
# to select rows from a specific Horse: df.loc[df['Horse ID']==0]
# https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values
df_slc_horses = df_all_horses.loc[df_all_horses.loc[:,'horseID'].isin(list_horseIDs_for_kde),:]
df_other_horses = df_all_horses.loc[~df_all_horses.loc[:,'horseID'].isin(list_horseIDs_for_kde),:]

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
    df.loc[:,'fraction_vis_kpts'] = frac_of_vis_kpts_per_frame

# %%
#################################################################################
## Select only frames in which over 90% of kpts are visible

# for horses selected for kde: 0.9
bdpts_per_frame_XY_slc_horses_valid = \
    bdpts_per_frame_XY_slc_horses[df_slc_horses['fraction_vis_kpts'] >= 0.9] # (4801, 22, 2)

# for horses for evaluation: 1
bdpts_per_frame_XY_other_horses_fully_viz = \
    bdpts_per_frame_XY_other_horses[df_other_horses['fraction_vis_kpts'] == 1] 


# plot how many frames discarded from set candidate to kde...
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
## Compute limbs lengths: pairwise distances between kpts for selected frames with valid poses
# Replace missing data with mean limb length
# (n selected frames, nchoosek(22,2) )
pairwise_dists_per_frame= np.vstack([pdist(data, "euclidean") \
                                        for data in bdpts_per_frame_XY_slc_horses_valid]) #(4801, 231) # for each frame, pass array of sorted keypoints # are these all in the same order? I guess so if data is 'sorted'
# replace missing data with mean
mu = np.nanmean(pairwise_dists_per_frame, axis=0) # mean limb length over all frames
missing = np.isnan(pairwise_dists_per_frame)
pairwise_dists_per_frame_no_nans = np.where(missing, mu, pairwise_dists_per_frame)


plt.matshow(pairwise_dists_per_frame_no_nans)
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

# PCA to get rid of multi-collinearity: 
# https://stats.stackexchange.com/questions/70899/what-correlation-makes-a-matrix-singular-and-what-are-implications-of-singularit
# https://stats.stackexchange.com/questions/142690/what-is-the-relation-between-singular-correlation-matrix-and-pca


# %%
#####################################################
## Estimate pdf using kernel density estimation
# Kernel density estimation is a way to estimate the probability density function (PDF) of a random variable 
# in a non-parametric way. gaussian_kde works for both uni-variate and multi-variate data. It includes automatic 
# bandwidth determination. The estimation works best for a unimodal distribution; bimodal or multi-modal distributions 
# tend to be oversmoothed.
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html 

kde_slc_horses = gaussian_kde(pairwise_dists_per_frame_no_nans.T) # if a 2D array input should be # dimensions, # data
kde_slc_horses.mean = mu

## evaluate gives 0?
# a = kde_slc_horses.resample(size=1)
# kde_slc_horses.pdf(a) # 0?? bc values are large?

# - kde.dataset = input data
# - kde.d = dimensions of the space (variables)
# - kde.n = number of datapoints
# - kde.neff =  effective number of datapoints
# - kde.factor = bandwidth factor
# - kde._data_covariance = covariance matrix of the input data
# - kde.covariance = The covariance matrix of dataset, scaled by the calculated bandwidth


# %%
##################################################################################
### For set of remaining horses: compute Mahalanobis distance

## Compute pairwise distances for remaining horses in frames with all kpts viz
# (n frames)
pairwise_dists_per_frame_other_horses = \
    np.asarray([pdist(arr, metric="euclidean") 
                for arr in bdpts_per_frame_XY_other_horses_fully_viz]) #bdpts_sel_frames_XY_other_horses])


## Compute Mahalanobis distance from 'other' poses to kde distribution
# why not the same as Jessy's method? :?
d_mahal = np.zeros((pairwise_dists_per_frame_other_horses.shape[0],1))
for i,d in enumerate(d_mahal):
    d_mahal[i,:] = mahalanobis(pairwise_dists_per_frame_other_horses[i,:],
                               kde_slc_horses.mean,
                               kde_slc_horses._data_inv_cov) #inv_cov? # kde_slc_horses._data_inv_cov


## Compute xth percentile (percentiles, inverse of cdf: ppf(q, df, loc=0, scale=1))
# d_mahal**2 follows X**2 distrib with dof = n of dimensions
# for points within distribution: d**2 follows chi2 distribution?
sq_d_mahal_percentile = chi2.ppf(chi2_percentile, 
                                 pairwise_dists_per_frame_other_horses.shape[1]) 
                                 # loc=0? scale=1? stats.ncx2?

# %%
############
## Compute Mahalanobis distance from points within distribution to kde distrbution to estimate the value for outliers
d_mahal_within = np.zeros((pairwise_dists_per_frame_no_nans.shape[0],1))
for i,d in enumerate(d_mahal_within):
    d_mahal_within[i,:] = mahalanobis(pairwise_dists_per_frame_no_nans[i,:],
                                      kde_slc_horses.mean, # np.mean(pairwise_dists_per_frame_no_nans, axis=0), #
                                      kde_slc_horses.inv_cov)  # _data_inv_cov
sq_d_mahal_percentile_within = chi2.ppf(0.05, 
                                        pairwise_dists_per_frame_no_nans.shape[1], # dof
                                        loc = pairwise_dists_per_frame_no_nans.shape[1],
                                        scale = 1) # loc=0? scale=1? stats.ncx2?
                                       
print(np.count_nonzero(d_mahal_within**2 < sq_d_mahal_percentile_within)/len(d_mahal_within))

plt.hist(d_mahal_within**2)
plt.show()
############

# %%
df = pairwise_dists_per_frame_no_nans.shape[1]
mean, var, skew, kurt = chi2.stats(df, 
                                   moments='mvsk')
x = np.linspace(chi2.ppf(0.01, df), #scale=chi2_scale,loc=-1500),
                chi2.ppf(0.99, df),100) #,scale=chi2_scale,loc=-1500), 100)

plt.plot(x, chi2.pdf(x, df),
        'r-', lw=5, alpha=0.6, label='chi2 pdf')          
# plt.hist(d_mahal_within**2, 
#          density=True, histtype='stepfilled', alpha=0.2)     
plt.show()

# %%
df = pairwise_dists_per_frame_no_nans.shape[1]
mean, var, skew, kurt = ncx2.stats(df, 
                                   np.sum(kde_slc_horses.mean**2), 
                                   moments='mvsk')
x = np.linspace(ncx2.ppf(0.01, df, np.sum(kde_slc_horses.mean**2)), #scale=chi2_scale,loc=-1500),
                ncx2.ppf(0.99, df, np.sum(kde_slc_horses.mean**2)),100) #,scale=chi2_scale,loc=-1500), 100)

plt.plot(x, ncx2.pdf(x, df, np.sum(kde_slc_horses.mean**2)),
        'r-', lw=5, alpha=0.6, label='chi2 pdf')          
# plt.hist(d_mahal_within**2, 
#          density=True, histtype='stepfilled', alpha=0.2)     
plt.show()
# %%
############
# ncx2
# d_mahal_within = np.zeros((pairwise_dists_per_frame_no_nans.shape[0],1))
# for i,d in enumerate(d_mahal_within):
#     d_mahal_within[i,:] = mahalanobis(pairwise_dists_per_frame_no_nans[i,:],
#                                       kde_slc_horses.mean, # np.mean(pairwise_dists_per_frame_no_nans, axis=0), #
#                                       kde_slc_horses.inv_cov)  # _data_inv_cov
# sq_d_mahal_percentile_within = ncx2.ppf(0.5, 
#                                         pairwise_dists_per_frame_no_nans.shape[1],
#                                         np.sqrt(np.sum(kde_slc_horses.mean**2)),
#                                         scale = 1000) # loc=0? scale=1? stats.ncx2?
                                       
# np.count_nonzero(d_mahal_within**2 < sq_d_mahal_percentile_within)/len(d_mahal_within)
# ############


# plot distance per sample frame
plt.plot(d_mahal)
plt.ylim([0,100])
plt.xlabel('frame ID')
plt.ylabel('Mahalanobis distance d (px)')
plt.hlines(sqrt(sq_d_mahal_percentile),
           0,len(d_mahal),
           'r')
plt.show()

plt.plot(d_mahal[:1300]**2)
plt.show()
plt.hist(d_mahal[:1300]**2)
plt.vlines(sq_d_mahal_percentile,
           0,max(d_mahal[:1300]**2),
           'r')
# plt.xlim([0,0.2e06])
plt.show()
# show a few frames from those above xth percentile



# show a few frames within xth percentile


### ncx2 instead of chi2?
# For example, the standard (central) chi-squared distribution is the 
# distribution of a sum of squared independent standard normal distributions, 
# i.e., normal distributions with mean 0, variance 1. 
# The noncentral chi-squared distribution generalizes this to normal 
# distributions with arbitrary mean and variance.
# https://en.wikipedia.org/wiki/Noncentral_distribution 
# - If each of the elements in the sum of squares (pairwise_dists_per_frame_other_horses)
#   can be considered a normally distributed r.v with mean 0, variance 1, then the 
#   sum of their squares (~ d_mahal**2) follows a standard (central) chi-squared distribution
# - But if each of the elements in pairwise_dists_per_frame_other_horses follows a 
#   normal distribution with arbitrary mean and variance, the noncentral chi-squared 
#   distribution should be used?

# %%
###################################################
## Express in terms of prob
# prob of the random variable d**2 being above the observed value?
# d is Mahalanobis distance, d**2 follows X**2 distribution
proba = np.zeros(d_mahal.shape)
for i,d in enumerate(proba):
    
    # COMPUTE p1=1-CDF.CHISQ(d-squared,nobsvar).
    # https://stats.stackexchange.com/questions/28593/mahalanobis-distance-distribution-of-multivariate-normally-distributed-points
    proba[i,:] = 1 - chi2.cdf(d_mahal[i,:]**2, 
                              pairwise_dists_per_frame_other_horses.shape[1]) 
   

# plot
plt.plot(proba,
         '.-')
plt.xlabel('frame ID')
plt.ylabel('prob of d**2 being above the observed value')
plt.hlines(chi2_percentile,
           0,len(proba),
           'r')

# %% 
########################################
########################################
# Jessy's approach?

# dev_sq_dists = pairwise_sq_dists_per_frame_other_horses - kde_slc_horses.mean # (1582, 231)
# mahal = sqrt(np.sum(((dev_sq_dists @ kde_slc_horses.inv_cov) * dev_sq_dists), axis=-1)) # factor * sqrt(np.sum((dot * dists), axis=-1))
# proba = 1 - chi2.cdf(mahal, np.sum(~mask))



# %%
#####################################################################################
# Calculate Mahalanobis distance between selected pose and reference distribution 
# (method of Assembler)
# compute pairwise distances in selected skeleton and the deviation of the vector wrt to the mean? 
# # It is a multi-dimensional generalization of the idea of measuring how many standard deviations away P is from the mean of D.
# #  https://en.wikipedia.org/wiki/Mahalanobis_distance 
# dists = pairwise_sq_dists_per_frame_other_horses - kde_slc_horses.mean #assembly.calc_pairwise_distances() - kde_slc_horses.mean #self._kde.mean
# mask = np.isnan(dists) # slc nans

# # Deal with nans
# if nan_policy_mahalanobis == "little":
#     inds = np.flatnonzero(~mask) # Return indices that are non-zero in the flattened version of a.
#     dists = dists[inds] #keep only those that are not nan
#     inv_cov = kde_slc_horses.inv_cov[np.ix_(inds, inds)] # self._kde.inv_cov[np.ix_(inds, inds)] # Using ix_ one can quickly construct index arrays that will index the cross product.
#     # Correct distance to account for missing observations
#     factor = kde_slc_horses.d / len(inds)
# else:
#     # Alternatively, reduce contribution of missing values to the Mahalanobis
#     # distance to zero by substituting the corresponding means.
#     dists[mask] = 0 # set distance of nans to 0
#     mask.fill(False) # sets all the mask to false? (as if no nans)
#     inv_cov = kde_slc_horses.inv_cov
#     factor = 1

# # compute Mahalanobis distance and prob
# dot = dists @ inv_cov # conventional matrix multiplication
# mahal = factor * sqrt(np.sum(((dists @ inv_cov) * dists), axis=-1))  # mean mahal distance?# factor * sqrt(np.sum((dot * dists), axis=-1))
# proba = 1 - chi2.cdf(mahal, np.sum(~mask)) #---- i think it should be mahal**2?
# print(proba)


# %%
# %%
###############################################################
# kde example

# def measure(n):
#     "Measurement model, return two coupled measurements."
#     x1 = np.random.normal(size=n)
#     x2 = np.random.normal(scale=0.5, size=n)
#     return x1+x2, x1-x2

# m1, m2 = measure(5000)
# xmin = m1.min()
# xmax = m1.max()
# ymin = m2.min()
# ymax = m2.max()

# values = np.vstack([m1, m2])
# kernel = gaussian_kde(values) # if a 2D array input should be # dimensions, # data

# X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
# positions = np.vstack([X.ravel(), Y.ravel()])
# Z = np.reshape(kernel(positions).T, X.shape)

# plt.imshow(np.rot90(Z), 
#           cmap=plt.cm.gist_earth_r,#  vmin=0.0,vmax=1.0,
#           extent=[xmin, xmax, ymin, ymax])
# plt.plot(m1, m2, 'k.', markersize=2)
# plt.xlim([xmin, xmax])
# plt.ylim([ymin, ymax])
# plt.colorbar()
# plt.show()

# print(kernel.evaluate(np.mean(values,axis=1)))
# # print(np.linalg.det(kernel.covariance))
# # print(np.linalg.det(kernel._data_covariance))

# # %%
# int_box = kernel.integrate_box(np.asarray([[-0.02579673, -0.00365534]]),
#                                np.asarray([[-0.02579673, -0.00365534]]))   
# print(kernel.evaluate(np.mean(values,axis=1)))
# print(int_box) # I would expect this to be close to evaluate output and its not?

# int_box = kernel.integrate_box(np.min(values,axis=1),
#                                np.max(values,axis=1))   

# print(int_box)