# %%
from cgi import test
import os
import glob
import pandas as pd

config_path = "/home/sabrina/stinkbugs-DLC-TO -PRESENT-19_07_22/config.yaml"
dirs_in_project = os.listdir(str(os.path.dirname(config_path)))
modelprefixes = []

for directory in dirs_in_project:
    if directory.startswith("data_augm_"):
        modelprefixes.append(directory)
modelprefixes.sort()
train_dict = {}
test_dict = {}
for modelprefix in modelprefixes:
    model_prefix = ''.join(['/home/sabrina/stinkbugs-DLC-TO -PRESENT-19_07_22/', modelprefix]) # modelprefix_pre = aug_
    aug_project_path = os.path.join(model_prefix, 'evaluation-results/iteration-1/') 
    #print(aug_project_path)
    csv_file = glob.glob(str(aug_project_path)+'*.csv')
    df1 = pd.read_csv(csv_file[0])
    train = list(df1[df1['Training iterations:'] == df1['Training iterations:'].max()].sort_values(by=['Shuffle number'])[' Train error(px)'])
    testt = list(df1[df1['Training iterations:'] == df1['Training iterations:'].max()].sort_values(by=['Shuffle number'])[' Test error(px)'])
    split_nesli = modelprefix.split('_')[-1]
    #if model_prefix.split('_')[-2].isdigit():
    #    split_nesli = modelprefix.split('_')[-1]
    #else:
    #    split_nesli = modelprefix.split('_')[-2] +' ' + modelprefix.split('_')[-1]
    train_dict[split_nesli] = train
    test_dict[split_nesli] = testt
# %%
sns.set_context("paper", rc={"font.size":12,"axes.titlesize":16,"axes.labelsize":12})  
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.DataFrame.from_dict(train_dict, orient='index')
df.index.rename('Model', inplace=True)

stacked = df.stack().reset_index()
stacked.rename(columns={'level_1': 'Shuffle', 0: 'Train Error Value (px)'}, inplace=True)

sns.swarmplot(data=stacked, x='Model', y='Train Error Value (px)', hue='Shuffle',size = 10)
plt.legend(title = 'Shuffle',bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xticks(rotation=45,fontsize = 10)
plt.ylim((2.3,3.25))
plt.grid()
plt.savefig('train.png', dpi=500, bbox_inches='tight')
plt.show()

# %%
df = pd.DataFrame.from_dict(test_dict, orient='index')
df.index.rename('Model', inplace=True)

stacked = df.stack().reset_index()
stacked.rename(columns={'level_1': 'Shuffle', 0: 'Test Error Value (px)'}, inplace=True)

sns.swarmplot(data=stacked, x='Model', y='Test Error Value (px)', hue='Shuffle',size = 10)
plt.legend( title="Shuffle", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xticks(rotation=45,fontsize = 10)
plt.ylim((2.3,3.25))
plt.grid()
plt.savefig('test.png',dpi=500, bbox_inches='tight')
plt.show()
#plt.savefig('./test.png', dpi=500, bbox_inches='tight')
# %%
