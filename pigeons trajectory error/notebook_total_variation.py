# %%
import deeplabcut
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
# %%
##########################################
# Input data
path_h5_baseline = "/home/neslihan/foraging-F-2020-11-24/data_augm_baseline/new_videos/pigeon_104DLC_resnet50_foragingNov24shuffle1_1000000.h5"
path_h5_grayscale = "/home/neslihan/foraging-F-2020-11-24/data_augm_grayscale/new_videos/pigeon_104DLC_resnet50_foragingNov24shuffle1_1000000.h5"


img_h_w = [960,1280] # image size (eventually get from data)
# %%
##########################################
# Read dataframes
df_baseline = pd.read_hdf(path_h5_baseline)
df_grayscale = pd.read_hdf(path_h5_grayscale)

df_baseline = df_baseline.droplevel('scorer',axis=1)
df_grayscale = df_grayscale.droplevel('scorer',axis=1)

#%%
##########################################
# Get list of bodyparts
list_bodyparts = [x for x,y in df_baseline.columns]
list_bodyparts = list(set(list_bodyparts))
list_bodyparts.sort()

list_bodyparts_to_analyse = [el for el in list_bodyparts \
                             if el not in ['a1','a2','a3','a4','b1','b2','b3','b4']]

# %%
list_colors_xy = ['tab:blue', 'tab:orange']
# list_colors_baseline_grayscale = ['tab:blue', 'k']
list_str_xy = ['x', 'y']
list_styles_model = ['-',':']

# initialise dict for total-variation results
total_variation_dict = dict()
# total_variation_dict = {'baseline':dict(),
#                         'grayscale':dict()}

flag_plot_per_bdprt = True
##########################################
# loop thru bodyparts to analyse
for bp in list_bodyparts_to_analyse: #['head']: #list_bodyparts:

    # Get bodyparts coordinates
    bp_xy_baseline_np = df_baseline[bp][['x','y']].to_numpy()
    bp_xy_grayscale_np = df_grayscale[bp][['x','y']].to_numpy()
    dict_bp_xy_per_model = {'baseline': bp_xy_baseline_np,
                            'grayscale': bp_xy_grayscale_np}



    # initialise figures
    # fig1,ax1_lst = plt.subplots(1,len(dict_bp_xy_per_model))
    # fig1.suptitle(bp)
    fig2,ax2_lst = plt.subplots(1,len(dict_bp_xy_per_model)) 
    fig2.suptitle(bp)
    fig3,ax3_lst = plt.subplots(1,bp_xy_baseline_np.shape[1]) 
    fig3.suptitle(bp)

    total_variation_dict[(bp,'x')] = []
    total_variation_dict[(bp,'y')] = []
    # loop thru models (baseline or grayscale)
    for i,model_str in enumerate(dict_bp_xy_per_model.keys()):

        np_coords_array = dict_bp_xy_per_model[model_str]

        # Compute total variation per bdprt for this model
        abs_diff_xy_coords = np.abs(np.diff(np_coords_array,axis=0))
        cumsum_abs_diff_xy_coords = np.cumsum(abs_diff_xy_coords,axis=0)   
        total_variation_dict[(bp,'x')].append(cumsum_abs_diff_xy_coords[-1,0])
        total_variation_dict[(bp,'y')].append(cumsum_abs_diff_xy_coords[-1,1])
        # total_variation_dict[(model_str,bp,'x')] = cumsum_abs_diff_xy_coords[-1,0]
        # total_variation_dict[(model_str,bp,'y')] = cumsum_abs_diff_xy_coords[-1,1]
        # total_variation_dict[model_str].update({bp:{'x': cumsum_abs_diff_xy_coords[-1,0],
        #                                             'y': cumsum_abs_diff_xy_coords[-1,1]}})

        # Compute median absolute deviation
        # np.median(np.absolute(x - np.median(x)))
         
        if flag_plot_per_bdprt:
            ### Fig trajectory in xy space       
            # sc=ax1_lst[i].scatter(np_coords_array[:,0],
            #                     np_coords_array[:,1],
            #                     c=list(range(np_coords_array.shape[0])),
            #                     s=2)
            # ax1_lst[i].set_xlim([0,img_h_w[1]])
            # ax1_lst[i].set_ylim([0,img_h_w[0]])
            # ax1_lst[i].set_xlabel('x (px)')
            # ax1_lst[i].set_ylabel('y (px)')
            # ax1_lst[i].invert_yaxis()
            # ax1_lst[i].set_title('{}'.format(model_str))
            # if i+1 == 2:
            #     # ax1_lst[i].legend(bbox_to_anchor=(1.1, 1.05))
            #     plt.colorbar(sc, ax=ax1_lst[i])

            ### Fig x,y-coord against frame
            for k in range(np_coords_array.shape[1]):
                if model_str == 'baseline':
                    col = 'k'
                else:
                    col = list_colors_xy[k]
                ax2_lst[k].plot(np_coords_array[:,k],
                                label=model_str,
                                color=col,
                                linestyle='-')
                ax2_lst[i].set_xlim([0,np_coords_array.shape[0]])
                ax2_lst[k].set_xlabel('frame')
                ax2_lst[k].set_title('{}-coord'.format(list_str_xy[k]))
                
            ax2_lst[k].legend(bbox_to_anchor=(1.1, 1.05))

            ### Fig total variation
            for k in range(np_coords_array.shape[1]):
                ax3_lst[k].plot(cumsum_abs_diff_xy_coords[:,k],
                                label=model_str,
                                color=list_colors_xy[k],
                                linestyle=list_styles_model[i])
                ax3_lst[k].set_title('{} {}-coord'.format(bp,list_str_xy[k]))
                ax3_lst[k].set_xlabel('frame')
                ax3_lst[k].set_ylabel('cum sum abs diffs (px)')

                # if cumsum_abs_diff_xy_coords[-1,k] > TV_prev:
                #     fweight = 'bold'
                # else:
                #     fweight = 'normal'
                ax3_lst[k].text(0.5,-(0.2 + i*0.1),
                                'TV {} = {:.2f} px'.format(model_str,cumsum_abs_diff_xy_coords[-1,k]),#   weight=fweight,
                                horizontalalignment='center',
                                verticalalignment='center', 
                                transform=ax3_lst[k].transAxes)
                
            ax3_lst[k].legend(bbox_to_anchor=(1.1, 1.05))

            
            ### Fig moving window mu-over-sigme
            
plt.show()

# %%
##########################################################
# Plot TV per bodypart

df_TV_per_bdprt = pd.DataFrame(total_variation_dict, 
                               index=['baseline', 'grayscale'])
df_TV_per_bdprt.columns.names = ['bodyparts','coords']
df_TV_per_bdprt.index.names = ['model']

# prepare for seaborn: remove multilevel columns
df_to_plot = df_TV_per_bdprt.stack(df_TV_per_bdprt.columns.names)\
                            .reset_index().rename(columns={0: 'TV'})
for c in ['x','y']:
    plt.figure(figsize=(12,10))
    sns.swarmplot(data=df_to_plot[df_to_plot['coords']==c],
                    x='bodyparts',
                    y='TV',
                    hue='model',
                    size = 10)
    # sns.lineplot(data=df_to_plot[df_to_plot['coords']==c],
    #                 x='bodyparts',
    #                 y='TV',
    #                 hue='model',
    #                 size = 10)
    plt.title('{}-coord [zoom]'.format(c))
    plt.xticks(rotation = 45) 
    plt.ylabel('total variation (px)')
    # plt.grid(which='minor', axis='y', linestyle='solid', color='black', alpha=0.2)
    plt.grid(which='major', axis='y', linestyle='solid', color='black', alpha=0.5)
    plt.ylim([0,500_000])

for c in ['x','y']:
    plt.figure(figsize=(12,10))
    sns.swarmplot(data=df_to_plot[df_to_plot['coords']==c],
                    x='bodyparts',
                    y='TV',
                    hue='model',
                    size = 10)
    # sns.lineplot(data=df_to_plot[df_to_plot['coords']==c],
    #                 x='bodyparts',
    #                 y='TV',
    #                 hue='model',
    #                 size = 10)
    plt.title('{}-coord [zoom]'.format(c))
    plt.ylabel('total variation (px)')
    plt.xticks(rotation = 45) 
    # plt.grid(which='minor', axis='y', linestyle='solid', color='black', alpha=0.2)
    plt.grid(which='major', axis='y', linestyle='solid', color='black', alpha=0.5)
    plt.ylim([0,80_000])

# %%
##########################################################
# Compute paired test
