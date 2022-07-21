from scipy.spatial import ConvexHull, convex_hull_plot_2d

import matplotlib.pyplot as plt
import numpy as np
rng = np.random.default_rng()
points = rng.random((30, 2))   # 30 random points in 2-D



hull = ConvexHull(points)
plt.plot(points[:,0], points[:,1], 'o')


for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
# %%
hull_indices = np.unique(hull.simplices.flat)
hull_pts = points[hull_indices, :]


def PolyArea(x,y): #https://en.wikipedia.org/wiki/Shoelace_formula
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

area_pol = PolyArea(hull_pts[:,0],hull_pts[:,1])
print('The sqrt of area is --->    ' + str(np.sqrt(area_pol)))
