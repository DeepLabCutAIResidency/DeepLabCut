'''
From Assembler and  Assembly class at: DeepLabCut/deeplabcut/pose_estimation_tensorflow/lib/inferenceutils.py

'''
# %%
##############################################################
## Imports
import numpy as np
import warnings
import pandas as pd
from math import sqrt, erf
import os
import random
import matplotlib.pyplot as plt
import cv2

# from scipy.optimize import linear_sum_assignment
# from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, mahalanobis #, cdist
# from scipy.special import softmax
from scipy.stats import gaussian_kde, chi2, ncx2

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


import deeplabcut
# from deeplabcut.pose_estimation_tensorflow.config import load_config
# from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils

# %%
#################################################################
## Input data 

# Labelled data
project_dir = '/home/sofia/datasets/Horses-Byron-2019-05-08'
labelled_data_h5file = \
    os.path.join(project_dir,'training-datasets/iteration-0/UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5') # h5 file

# list of HorseIDs to use in distribution estimate
list_horseIDs_for_kde = [*range(15)]

# normalising distance in pixels
# dist_diag_in_px = sqrt(288**2 + 162**2) # for normalisation before PCA

# parameter to determine number of PCA components
min_variance_retained = 0.95 #0.95

# mahalanobis
chi2_percentile = 0.95

# seed:
random.seed(3)

# %%
##########################################################
## Read all labelled data 
df_all_horses = pd.read_hdf(labelled_data_h5file)

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


# %%
#########################################################################
# Add fraction of kpts visible per frame

## Split selected horses (and other horses)
# to select rows from a specific Horse: df.loc[df['Horse ID']==0]
# https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values
df_slc_horses = df_all_horses.loc[df_all_horses.loc[:,'horseID'].isin(list_horseIDs_for_kde),:]

## Get matrix of bdprts  coords  per frame 
# Reshape joint data to matrix of (n_frames, n_bodyparts, 2) 
bdpts_per_frame_XY_slc_horses = df_slc_horses.drop('horseID',axis=1,level=0).to_numpy().reshape((-1, n_bpts, 2)) # (8114, 22, 2)

## Add fraction of kpts visible per frame
# Compute how complete each frame is (all kpts =1) 
for bdpts_arr,df in zip([bdpts_per_frame_XY_slc_horses],
                        [df_slc_horses]):
    frac_of_vis_kpts_per_frame = np.mean(~np.isnan(bdpts_arr), axis=(1, 2)) 

    # add to dataframe
    df.loc[:,'fraction_vis_kpts'] = frac_of_vis_kpts_per_frame

# %%
##############################################################################
## Compute median nose-eye distance for normalisation
# nose_xy = df_slc_horses.iloc[:,df_slc_horses.columns.get_level_values(1)=='Nose'].to_numpy()
# eye_xy = df_slc_horses.iloc[:,df_slc_horses.columns.get_level_values(1)=='Eye'].to_numpy()

# dist_nose2eye = [np.linalg.norm(nose_xy_i - eye_xy_i) 
#                     for (nose_xy_i,eye_xy_i) in zip(nose_xy,eye_xy)]
# median_dist_nose2eye_slc_horses = np.nanmedian(dist_nose2eye) # 17.49

# plt.plot(dist_nose2eye)
# plt.hlines(median_dist_nose2eye_slc_horses,
#           0,len(dist_nose2eye),'r')
# plt.ylim([0,45])
# plt.xlabel('frame ID')
# plt.ylabel('nose2eye distance (px)')
# plt.show()

# # all labelled data
# nose_xy = df_all_horses.iloc[:,df_all_horses.columns.get_level_values(1)=='Nose'].to_numpy()
# eye_xy = df_all_horses.iloc[:,df_all_horses.columns.get_level_values(1)=='Eye'].to_numpy()

# dist_nose2eye = [np.linalg.norm(nose_xy_i - eye_xy_i) 
#                     for (nose_xy_i,eye_xy_i) in zip(nose_xy,eye_xy)]

# median_all_horses = np.nanmedian(dist_nose2eye) #16.57

# plt.plot(dist_nose2eye)
# plt.hlines(median_all_horses,
#           0,len(dist_nose2eye),'r')
# plt.ylim([0,45])
# plt.xlabel('frame ID')
# plt.ylabel('nose2eye distance (px)')
# %%
#################################################################################
## Select only frames from slc subset in which over 90% of kpts are visible

# for horses selected for kde: 0.9
df_slc_horses_valid = df_slc_horses.loc[df_slc_horses['fraction_vis_kpts'] >= 0.9]
bdpts_per_frame_XY_slc_horses_valid = \
    bdpts_per_frame_XY_slc_horses[df_slc_horses['fraction_vis_kpts'] >= 0.9] # (4801, 22, 2)

# plot how many frames discarded from set candidate to kde...
print(len(df_slc_horses_valid)/len(df_slc_horses))

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

# normalise by median nose2eye distance
# pairwise_dists_per_frame_no_nans_norm = \
#     pairwise_dists_per_frame_no_nans/dist_diag_in_px #median_dist_nose2eye_slc_horses


plt.matshow(pairwise_dists_per_frame_no_nans)
plt.legend()
plt.xlabel('sorted pairs of kpts')
plt.ylabel('selected frames')
plt.colorbar()
plt.title('pairwise distances (px)')
plt.show()
# %%
######################################################################
## Compute PCA components here? (and then pass that to Gaussian KDE?)
# https://towardsdatascience.com/principal-component-analysis-pca-with-scikit-learn-1e84a0c731b0 

# PCA to get rid of multi-collinearity: 
# https://stats.stackexchange.com/questions/70899/what-correlation-makes-a-matrix-singular-and-what-are-implications-of-singularit
# https://stats.stackexchange.com/questions/142690/what-is-the-relation-between-singular-correlation-matrix-and-pca

# scale each feature for mean=0 and var=1
scaler = StandardScaler()
scaler.fit(pairwise_dists_per_frame_no_nans) # calculate the mean and standard deviation for each variable in the dataset
pairwise_dists_per_frame_no_nans_scaled = scaler.transform(pairwise_dists_per_frame_no_nans)


# #---------------------------------------------
# pairwise_dists_per_frame_transformed = pairwise_dists_per_frame_no_nans_scaled

# #---------------------------------------------


# pca 
pca_obj = PCA(min_variance_retained)
pca_obj.fit(pairwise_dists_per_frame_no_nans_scaled)
pairwise_dists_per_frame_transformed = pca_obj.transform(pairwise_dists_per_frame_no_nans_scaled)


print(pca_obj.n_components_)
print(pca_obj.n_components)


# bar plot
plt.bar(np.arange(1,pca_obj.n_components_+1),
        pca_obj.explained_variance_ratio_)
plt.xlim([1,pca_obj.n_components_])
plt.ylim([0,1])
plt.xlabel('PCA component')
plt.ylabel('fraction of variance explained')
plt.show()

# cum plot
plt.plot(np.arange(1,pca_obj.n_components_+1),
         np.cumsum(pca_obj.explained_variance_ratio_),
        'r.-')
plt.hlines(0.95,0,pca_obj.n_components_+1,'k',linestyle=':')
plt.xlim([0,pca_obj.n_components_+1])
plt.ylim([0.7,1])
plt.xlabel('PCA component')
plt.ylabel('fraction of variance explained')
plt.show()

# print min num of components to get explained variance above th
pca_idcs = np.where(np.cumsum(pca_obj.explained_variance_ratio_)>0.95)[0]
pca_min_components = pca_idcs+1
np.cumsum(pca_obj.explained_variance_ratio_)[pca_idcs]

# n_components
# copy
# whiten
# svd_solver
# tol
# iterated_power
# n_oversamples
# power_iteration_normalizer
# random_state
# n_features_in_
# _fit_svd_solver
# mean_
# noise_variance_
# n_samples_
# n_features_
# components_
# n_components_
# explained_variance_
# explained_variance_ratio_
# singular_values_


# %%
#####################################################
## Estimate pdf using kernel density estimation
# Kernel density estimation is a way to estimate the probability density function (PDF) of a random variable 
# in a non-parametric way. gaussian_kde works for both uni-variate and multi-variate data. It includes automatic 
# bandwidth determination. The estimation works best for a unimodal distribution; bimodal or multi-modal distributions 
# tend to be oversmoothed.
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html 



kde_slc_horses = gaussian_kde(pairwise_dists_per_frame_transformed.T) # if a 2D array input should be # dimensions, # data
kde_slc_horses.mean = np.mean(pairwise_dists_per_frame_transformed, axis=0)

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
##############################################################
## Compute Mahalanobis distance from points within distribution to kde distrbution to estimate the value for outliers
d_mahal_within = np.zeros((pairwise_dists_per_frame_transformed.shape[0],1))
for i,d in enumerate(d_mahal_within):
    d_mahal_within[i,:] = mahalanobis(pairwise_dists_per_frame_transformed[i,:],
                                      kde_slc_horses.mean, # np.mean
                                      kde_slc_horses.inv_cov)  # _data_inv_cov


# add to dataframe
df_slc_horses_valid['d_mahal'] = d_mahal_within

# compute 95th percentile
d_mahal_95th = np.percentile(d_mahal_within, chi2_percentile*100)

d_mahal_10_50_75_95th = list()
list_percentiles = [10, 50, 75, 95, 99.99]
idx_95 = int(np.nonzero(np.asarray(list_percentiles)==95)[0])
for l in list_percentiles:
    d_mahal_10_50_75_95th.append(np.percentile(d_mahal_within, l))


# plot histogram
(n,b,p) = plt.hist(d_mahal_within)
for d in d_mahal_10_50_75_95th:
    if d == d_mahal_10_50_75_95th[idx_95]:
        continue
    plt.vlines(d,0, 1.1*max(n),'b',linestyle=':') 
plt.vlines(d_mahal_10_50_75_95th[idx_95],0, 1.1*max(n),'r',linestyle=':') # 95
plt.xlabel(r'$d_{Mahal}$ (PCA space)')
plt.ylabel('count')
plt.show()

# plot per frame
plt.plot(d_mahal_within,'.-')
# plt.scatter(np.argwhere(np.any(missing,axis=1)),
#          d_mahal_within[np.argwhere(np.any(missing,axis=1)).squeeze()],
#          s=50, facecolors='k', edgecolors='k')
plt.hlines(d_mahal_10_50_75_95th[idx_95],0, len(d_mahal_within),'r',linestyle=':')
plt.xlabel('frame ID')
plt.ylabel(r'$d_{Mahal}$ (PCA space)')
plt.show()

# plot distance against fraction of frames
# plt.scatter(df_slc_horses_valid['d_mahal'],
#             df_slc_horses_valid['fraction_vis_kpts'],40,'b')
# plt.ylim([0.9,1.01])
# plt.vlines(d_mahal_th,0, 1.01,'r',linestyle=':')
# plt.xlabel(r'$d_{Mahal}$')
# plt.ylabel('fraction visible kpts')
###########################################################################
# %% Find idcs closest to percentiles

# compute closest sample to percentile
idcs_per_percentile = list()
for d in d_mahal_10_50_75_95th:
    idcs_per_percentile.append(np.argmin(np.abs(d_mahal_within - d)))

# plot image with keypoints
for j,idx in enumerate(idcs_per_percentile):
    image_path = os.path.join(project_dir,
                              df_slc_horses_valid.index[idx])

    # check
    print(d_mahal_within[idx])                 
    print(df_slc_horses_valid['d_mahal'][idx])            

    # plot image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # https://stackoverflow.com/questions/39316447/opencv-giving-wrong-color-to-colored-images-on-loading
    fig = plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.text(1, 0.975*image.shape[0], df_slc_horses_valid.index[idx], fontsize=12)
    
    
    # plot keypoints
    kpts_array = df_slc_horses_valid.iloc[idx,:].drop(['horseID','fraction_vis_kpts','d_mahal'],level=0)\
                .to_numpy().reshape(-1,2)

    plt.scatter(kpts_array[:,0],kpts_array[:,1],40,'r')
    plt.title('Percentile {:8.2f}th, d={:8.2f}'\
                .format(list_percentiles[j],
                        float(d_mahal_within[idx])))
    plt.text(0.8*image.shape[1], 0.975*image.shape[0], 
             'visible kpts:{}/{}'.format(len(kpts_array)-np.sum(np.isnan(kpts_array[:,0])),
                                         len(kpts_array)), fontsize=12)
###########################################################################
# %% Inspect frames in terms of d_mahal_within

# retreive frames with lowest and highest d_mahal
n_frames_to_slc = 3
idcs_d_mahal_sorted = np.argsort(d_mahal_within.squeeze()) # ascending

list_top_n_closest_frames = \
    df_slc_horses_valid.index.to_numpy()[idcs_d_mahal_sorted[:n_frames_to_slc]].squeeze().tolist()

list_top_n_furthest_frames = \
    df_slc_horses_valid.index.to_numpy()[idcs_d_mahal_sorted[::-1][:n_frames_to_slc]].squeeze().tolist()


print(set([x.split('/')[1] for x in df_slc_horses_valid.index]))


# %%
###########################################################################
# Plot images close to mean (in terms of d_Mahal)

# plot image with keypoints
for (el,idx) in zip(list_top_n_closest_frames,
                    idcs_d_mahal_sorted):
    image_path = os.path.join(project_dir,el)

    # plot image
    image = cv2.imread(image_path)
    fig = plt.figure(figsize=(10,10))
    plt.imshow(image)

    # plot keypoints
    kpts_array = df_slc_horses_valid[df_slc_horses_valid.index==el]\
                .drop('horseID',axis=1,level=0).drop('fraction_vis_kpts',axis=1,level=0)\
                .to_numpy().reshape(-1,2)

    plt.scatter(kpts_array[:,0],kpts_array[:,1],40,'r')
    plt.title('d = {}, visible kpts:{}/{}'.format(d_mahal_within[idx],
                                                  len(kpts_array)-np.sum(np.isnan(kpts_array[:,0])),
                                                  len(kpts_array)))


###########################################################################
# Plot images far to mean (in terms of d_Mahal)

# plot image with keypoints
for (el,idx) in zip(list_top_n_furthest_frames,
                    idcs_d_mahal_sorted[::-1]):
    image_path = os.path.join(project_dir,el)

    # plot image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # https://stackoverflow.com/questions/39316447/opencv-giving-wrong-color-to-colored-images-on-loading
    fig = plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.text(1, image.shape[0], image_path, fontsize=12)

    # plot keypoints
    kpts_array = df_slc_horses_valid[df_slc_horses_valid.index==el]\
                .drop('horseID',axis=1,level=0).drop('fraction_vis_kpts',axis=1,level=0)\
                .to_numpy().reshape(-1,2)

    plt.scatter(kpts_array[:,0],kpts_array[:,1],40,'r')
    plt.title('d = {}, visible kpts:{}/{}'.format(d_mahal_within[idx],
                                                  len(kpts_array)-np.sum(np.isnan(kpts_array[:,0])),
                                                  len(kpts_array)))

# %%
##############################
## Compute d_th such that 95% of samples are closer to mean than d_th value (in Mahalanobis distance)


# stats for standard chi-squared distrib with dof=n_dims
# chi2_mean, chi2_var, skew, kurt = chi2.stats(pairwise_dists_per_frame_transformed.shape[1], 
#                                              moments='mvsk',
#                                              loc=0,
#                                              scale=1)
#-----------------
# chi2_mean = 0
# chi2_var=1
#------------
# compute 95% percentile
sq_d_mahal_percentile_within = chi2.ppf(chi2_percentile, 
                                        pairwise_dists_per_frame_transformed.shape[1],
                                        loc=chi2_mean,
                                        scale=chi2_var) # loc=0? scale=1? stats.ncx2?

# compare percentile from chi-squared to 'empirical' percentile                                  
print('Exact fraction of samples under th: {}'.\
      format(np.count_nonzero(d_mahal_within**2 < sq_d_mahal_percentile_within)/len(d_mahal_within)))
print('Input percentile fraction: {}'.format(chi2_percentile))
print('-----------')
print('Exact percentile value: {}'.format(np.percentile(d_mahal_within**2, chi2_percentile*100)))
print('Percentile value from chi-squared: {}'.format(sq_d_mahal_percentile_within))

# plt.hist(d_mahal_within**2)
# plt.show()
############

# %%
#############################################################
#### Check chi-squared assumption for d**2
sq_d_mahal_hist_bin_width = 10 #25
### plot chi-squared distrib for this number of dimensions
x = np.linspace(chi2.ppf(0.01, 
                         pairwise_dists_per_frame_transformed.shape[1], 
                         loc=chi2_mean, scale=chi2_var), #scale=chi2_scale,loc=-1500),
                chi2.ppf(0.99, 
                          pairwise_dists_per_frame_transformed.shape[1], 
                          loc=chi2_mean, scale=chi2_var),100) #,scale=chi2_scale,loc=-1500), 100)
plt.plot(x, chi2.pdf(x, 
                     pairwise_dists_per_frame_transformed.shape[1], 
                     loc=chi2_mean, 
                     scale=chi2_var),
        'r-', lw=5, alpha=0.6, label='chi2 pdf')  
# plt.vlines(sq_d_mahal_percentile_within,
#           0, np.max(d_mahal_within**2)/10000,
#           'r',linestyle=':',alpha=0.6)

## plot histogram with data        
plt.hist(d_mahal_within**2, 
         np.arange(0,
                   np.round(np.max(d_mahal_within**2)/sq_d_mahal_hist_bin_width)*sq_d_mahal_hist_bin_width + sq_d_mahal_hist_bin_width,
                   sq_d_mahal_hist_bin_width),
         density=True, histtype='stepfilled', alpha=0.2)     
# plt.vlines(np.percentile(d_mahal_within**2, chi2_percentile*100),
#           0, np.max(d_mahal_within**2),
#           'b',linestyle=':',alpha=0.6)
plt.xlabel(r'$d_{Mahal}^2$')
plt.ylabel('normalised count')
plt.show()
 
# %%
#### ncx2
# x = np.linspace(ncx2.ppf(0.01, 
#                         pairwise_dists_per_frame_transformed.shape[1], 
#                         np.sum(np.mean(pairwise_dists_per_frame_transformed,axis=0)**2),#0 if pca
#                         loc=chi2_mean, scale=chi2_var), #scale=chi2_scale,loc=-1500),
#                 ncx2.ppf(0.99,
#                          pairwise_dists_per_frame_transformed.shape[1], 
#                           np.sum(np.mean(pairwise_dists_per_frame_transformed,axis=0)**2),
#                           loc=chi2_mean, scale=chi2_var),
#                 100) #,scale=chi2_scale,loc=-1500), 100)

# ### data
# plt.plot(x, ncx2.pdf(x, 
#                      pairwise_dists_per_frame_transformed.shape[1], 
#                      np.sum(np.mean(pairwise_dists_per_frame_transformed,axis=0)**2),
#                      loc=chi2_mean, scale=chi2_var),
#         'g-', lw=5, alpha=0.6, label='cnx2 pdf')      
   
# (h,bins,p)=plt.hist(d_mahal_within**2, 
#                     density=True, histtype='stepfilled', alpha=0.2)     
# plt.legend()
# plt.show()
# %%
