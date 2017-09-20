import numpy as np
import cv2
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

import skvideo.io
import skvideo.datasets
cap = skvideo.io.vread('/home/nsr/Documents/Test and Obstacles/VID_20170917_120056_1.mp4')
i=0
def colorext(img):
	
	img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
	lower_red = np.array([0,50,50])
	upper_red = np.array([10,255,255])
	mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# upper mask (170-180)
	lower_red = np.array([170,50,50])
	upper_red = np.array([180,255,255])
	mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join my masks
	mask = mask0+mask1

# set my output img to zero everywhere except my mask
	output_img = img.copy()
	output_img[np.where(mask==0)] = 0
	return output_img

frame=cap.take((6),axis=0)
frame=cv2.resize(frame,(600,600))


#while(True):
    
     #frame=cap.take((i),axis=0)
      #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     #frame =colorext(frame)
cv2.imshow('frame',frame)
    
     
cv2.waitKey(0) & 0xFF == ord('q')  
    

            

#cap.release()
#cv2.destroyAllWindows()
