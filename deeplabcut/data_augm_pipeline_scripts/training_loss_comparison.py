import os
import glob
from numpy import size
import pandas as pd
import matplotlib.pyplot as plt

config_path = "/home/sabrina/stinkbugs-DLC-TO -PRESENT-19_07_22/config.yaml" # path of the config file
dirs_in_project = os.listdir(str(os.path.dirname(config_path)))
modelprefixes = []

for directory in dirs_in_project:
    if directory.startswith("data_augm_"):
        modelprefixes.append(directory)
modelprefixes.sort()
my_dict = {}

for modelprefix in modelprefixes:
    model_prefix = ''.join(['/home/sabrina/stinkbugs-DLC-TO -PRESENT-19_07_22/', modelprefix]) # modelprefix_pre = aug_
    aug_project_path = os.path.join(model_prefix, 'dlc-models/iteration-1/') 

    csv_file = glob.glob(str(aug_project_path)+"/**/learning_stats.csv", recursive = True)
    header_list = ["number_of_iterations", "loss", "lr"]
    sh = 0
    for file in csv_file:
        df = pd.read_csv(file,names=header_list)

        if model_prefix.split('_')[-2].isdigit():
            name = modelprefix.split('_')[-1]
        else:
            name = modelprefix.split('_')[-2] +' ' + modelprefix.split('_')[-1]

        my_dict[name + str(sh) ] = list(df['loss']), list(df['number_of_iterations'])
        sh +=1

# All shuffles together: 
for i in my_dict.keys():

    plt.plot(my_dict[i][1],my_dict[i][0],'-',label = i)
    plt.xlabel('Num. Iterations')
    plt.ylabel('loss')
    plt.legend(title="augmentation type", bbox_to_anchor=(1.05, 1.05), loc=2, borderaxespad=0.)
plt.show()

# If we want to plot shuffles separetely:
fig, ax = plt.subplots(1,2, figsize = (12,8)) #only works with 2 shuffles

for i in my_dict.keys():
    if '0' in i:
        ax[0].plot(my_dict[i][1],my_dict[i][0],'-',label = i,linewidth = 3)
        ax[0].set_xlabel('Num. Iterations',fontsize = 18)
        ax[0].set_ylabel('loss',fontsize = 18)
        #ax[0].set_xlim()
        ax[0].legend(title="augmentation type", bbox_to_anchor=(1,1))
    else: 
        ax[1].plot(my_dict[i][1],my_dict[i][0],'-',label = i,linewidth = 3)
        ax[1].set_xlabel('Num. Iterations',fontsize = 18)
        #ax[1].set_ylabel('loss')
        ax[1].legend(title="augmentation type", bbox_to_anchor=(1.0, 1.0))

plt.show()
fig.savefig('./loss_comparison.png', dpi=500, bbox_inches='tight')

