# %%
import deeplabcut
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
path_h5_baseline = "/home/neslihan/foraging-F-2020-11-24/data_augm_baseline/new_videos/pigeon_104DLC_resnet50_foragingNov24shuffle1_1000000.h5"
path_h5_grayscale = "/home/neslihan/foraging-F-2020-11-24/data_augm_grayscale/new_videos/pigeon_104DLC_resnet50_foragingNov24shuffle1_1000000.h5"


img_h_w = [960,1280]
# %%
df_baseline = pd.read_hdf(path_h5_baseline)
df_grayscale = pd.read_hdf(path_h5_grayscale)
# %%
df_baseline = df_baseline.droplevel('scorer',axis=1)
df_grayscale = df_grayscale.droplevel('scorer',axis=1)

#%%
list_bodyparts = [x for x,y in df_baseline.columns]
list_bodyparts = list(set(list_bodyparts))
list_bodyparts.sort()
# list_bodyparts2 = list()
# for x,y in df_baseline.columns:
#     list_bodyparts2.append(x)
# %%

for bp in ['head']: #list_bodyparts:

    bp_xy_baseline_np = df_baseline[bp][['x','y']].to_numpy()
    bp_xy_grayscale_np = df_grayscale[bp][['x','y']].to_numpy()

    # plt.scatter(bp_xy_baseline_np)
    dict_np_per_model = {'baseline': bp_xy_baseline_np,
                        'grayscale': bp_xy_grayscale_np}

    # one figure per bdprt
    fig1,ax1_lst = plt.subplots(1,len(dict_np_per_model))
    fig2,ax2_lst = plt.subplots(1,len(dict_np_per_model)) 

    fig1.suptitle(bp)
    fig2.suptitle(bp)
    for i,model_str in enumerate(dict_np_per_model.keys()):

        np_coords_array = dict_np_per_model[model_str]
         
        # plot trajectory in xy space       
        ax1_lst[i].scatter(np_coords_array[:,0],
                            np_coords_array[:,1],
                            c=list(range(np_coords_array.shape[0])),
                            s=2,
                            label=bp)
        ax1_lst[i].set_xlim([0,img_h_w[1]])
        ax1_lst[i].set_ylim([0,img_h_w[0]])
        ax1_lst[i].set_xlabel('x (pixels)')
        ax1_lst[i].set_ylabel('y (pixels)')
        ax1_lst[i].invert_yaxis()
        ax1_lst[i].set_title('{}'.format(model_str))
        if i+1 == 2:
            ax1_lst[i].legend(bbox_to_anchor=(1.1, 1.05))
        # fig.colorbar(im2, cax=cax, orientation='vertical')

         # plot x-coord against frame
        ax2_lst[i].plot(np_coords_array[:,0],'-',
                        label='x coord')
        ax2_lst[i].plot(np_coords_array[:,1],'-',
                        label='y coord')
        ax2_lst[i].set_xlim([0,img_h_w[1]])
        ax2_lst[i].set_xlabel('frame')
        ax2_lst[i].set_ylabel('x,y-coord (pixels)')
        ax2_lst[i].set_title('{}'.format(model_str))
plt.show()

# %%
