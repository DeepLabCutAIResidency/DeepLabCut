# %%
import os
import glob
import re

# Absolute path of a file
pt1 = 'C:/Users/Sabrina/Desktop/DLC Res/frameSelection/week1/keyframes-HORSE/1/*.png'
pt2 = 'C:/Users/Sabrina/Desktop/DLC Res/frameSelection/week1/keyframes-HORSE/2/*.png'

file1 = glob.glob(pt1)
file2 = glob.glob(pt2)

img1 = [i.split('\\')[-1] for i in file1]
img2 = [i.split('\\')[-1] for i in file2]

# %%
# Renaming the file
for i in img2: 
    num = int(re.findall(r'\d+', i.split('.')[0])[0])
    numfix = int(num)
    if len(str(numfix)) ==1 : 
        os.rename('C:/Users/Sabrina/Desktop/DLC Res/frameSelection/week1/keyframes-HORSE/2/' +str(i), '000'+str(numfix)+'.png')
    if len(str(numfix)) ==2 : 
        os.rename('C:/Users/Sabrina/Desktop/DLC Res/frameSelection/week1/keyframes-HORSE/2/' +str(i), '00'+str(numfix)+'.png')
    if len(str(numfix)) ==3 : 
        os.rename('C:/Users/Sabrina/Desktop/DLC Res/frameSelection/week1/keyframes-HORSE/2/' +str(i), '0'+str(numfix)+'.png')
# %%
