import cv2
import numpy as np
import glob
import os
 
img_array = []

exp = './exp_20220314-145218'
jpg = '/*'

for filename in sorted(glob.glob(exp+jpg), key=os.path.getctime):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 

out = cv2.VideoWriter('{}.mp4'.format(exp),cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()