import cv2
import numpy as np
import glob
import re
path = 'C:/Users/Sabrina/Desktop/DLC Res/frameSelection/week1/BrownHorseinShadow/*.png'
frames = glob.glob(path)
frames.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])


img_array = []
for filename in frames:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()