import cv2
from tqdm import tqdm
import os
import time 
train='/home/nsr/Documents/dataset_car'
for img in tqdm(sorted(os.listdir(train))):
    time.sleep(0.1)
    path =os.path.join(train,img)
    img=cv2.imread(path)
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


''''
import numpy as np

a = np.array([32.49, -39.96,-3.86])
b = np.array([31.39, -39.28, -4.66])
c = np.array([31.14, -38.09,-4.49])

ba = a - b
bc = c - b

cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
angle = np.arccos(cosine_angle)

print np.degrees(angle)'''
