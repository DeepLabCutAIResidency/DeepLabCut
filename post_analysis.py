# %%
import deeplabcut
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
path_h5_baseline = "/home/neslihan/foraging-F-2020-11-24/data_augm_baseline/new_videos/pigeon_104DLC_resnet50_foragingNov24shuffle1_1000000.h5"
path_h5_grayscale = "/home/neslihan/foraging-F-2020-11-24/data_augm_grayscale/new_videos/pigeon_104DLC_resnet50_foragingNov24shuffle1_1000000.h5"

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

    for i,model_str in enumerate(dict_np_per_model.keys()):

        ax1 = plt.subplot(1,len(dict_np_per_model),i+1)

        np_coords_array = dict_np_per_model[model_str]


        # plot x-coord against frame
         
        # plot trajectory in xy space
        ax1.plot(np_coords_array[:,0],
                 np_coords_array[:,1],
                 '.',
                 label=bp)
        ax1.set_xlim([0,1280])
        ax1.set_ylim([0,960])
        ax1.set_xlabel('x (pixels)')
        ax1.set_ylabel('y (pixels)')
        ax1.invert_yaxis()
        ax1.set_title('{}'.format(model_str))
        if i+1 == 2:
            ax1.legend(bbox_to_anchor=(1.1, 1.05))


plt.show()
