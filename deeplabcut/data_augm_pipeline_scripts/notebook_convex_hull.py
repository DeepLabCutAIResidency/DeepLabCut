from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% Aux fns
def PolyArea(x,y): 
    # https://en.wikipedia.org/wiki/Shoelace_formula
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

# %% Read groundtruth data
human_labels_filepath ='/media/data/stinkbugs-DLC-2022-07-15/data_augm_00_baseline/training-datasets/iteration-1/UnaugmentedDataSet_stinkbugsJul15/CollectedData_DLC.h5' #'/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/training-datasets/iteration-0/UnaugmentedDataSet_sabris-mouseJul6/CollectedData_nirel.h5'
df_human = pd.read_hdf(human_labels_filepath)

df_human = df_human.droplevel('scorer',axis=1) #df_human['DLC'][:].iloc[0,:]
# %% Load image
rng = np.random.default_rng()
points = rng.random((30, 2))   # 30 random points in 2-D
plt.plot(points[:,0], points[:,1], 'o')

# %%  Compute convex hull
hull = ConvexHull(points)
hull_indices = np.unique(hull.simplices.flat)
hull_pts = points[hull_indices, :]

# for simplex in hull.simplices:
#     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

# %% Compute area
area_pol = PolyArea(hull_pts[:,0],
                    hull_pts[:,1])
print('The sqrt of area is --->    ' + str(np.sqrt(area_pol)))
