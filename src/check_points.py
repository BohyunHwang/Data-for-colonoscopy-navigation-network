import cv2
import os
from opt import parse_args
from sklearn.model_selection import train_test_split
from utils import read_data
import glob
import pandas as pd
import numpy as np

args = parse_args()

exp = 'Sequence12'
os.makedirs(os.path.join(args.model_name,'predictions','detections','{}'.format(exp)), exist_ok= True)
label = pd.read_csv(os.path.join(args.dataset,'labels',exp + '.csv'))
pred = np.load('./{}_pred.npy'.format(exp))


print(label, pred)

for l,p in zip(label.values, pred):
    x = min(max(p[0,0],-1),1)
    y = min(max(p[0,1],-1),1)
    img = cv2.imread(os.path.join(args.dataset, 'images', l[-1]))
    cv2.circle(img, (l[2],l[3]),10,(180,119,31),-1)
    cv2.circle(img,(int(x*640),int(y*480)), 10,(14,127,255), -1)
    
    filename = os.path.join(args.model_name, 'predictions', 'detections','{}'.format(exp), '{}'.format(l[1]))

    cv2.imwrite(filename, img)