import numpy as np
import cv2
from collections import deque
import numpy as np
import argparse
import imutils
import time
import cv2
from tqdm import tqdm
import os


##import pims
##v = pims.Video('/home/nsr/Documents/Test and Obstacles/VID_20170917_120056_2.mp4')

import av
v = av.open('/home/nsr/Documents/Test and Obstacles/VID_20170917_120056_2.mp4')
##import skvideo.io
##import skvideo.datasets
##cap = skvideo.io.vread('/home/nsr/Documents/Test and Obstacles/VID_20170917_120056_2.mp4')
##i=0
#blue
##greenLower = (100, 30, 3)
##greenUpper = (255, 140, 80)

#pink
greenLower = (90, 30, 150)
greenUpper = (200, 100, 255)


#pts = deque(maxlen=args["buffer"])
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



i=0

train='/home/nsr/Documents/dataset_car'
for img in tqdm(sorted(os.listdir(train))):
#while(True):
     time.sleep(0.1)
     path =os.path.join(train,img)
     frame=cv2.imread(path)
     #frame=cap.take((i),axis=0)
     #frame=v[i]
##     for packet in v.demux():
##            for frame in packet.decode():
##                if frame.type == 'video':
##                    img = frame.to_image()  # PIL/Pillow image
##                    frame = np.asarray(img)
     frame = imutils.resize(frame, width=600)
     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     #frame =colorext(frame)
     mask = cv2.inRange(frame, greenLower, greenUpper)
     mask = cv2.erode(mask, None, iterations=6)
     mask = cv2.dilate(mask, None, iterations=2)
     cv2.imshow('mask',mask)

     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		             cv2.CHAIN_APPROX_SIMPLE)[-2]
     center = None
     i+=1
     if len(cnts) > 0:
                
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                print x, y
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
                        cv2.circle(hsv, (int(x), int(y)), int(radius),
			(0, 255, 255), 2)
                        cv2.circle(hsv, center, 5, (0, 0, 255), -1)
     cv2.imshow('frame',hsv)
     if cv2.waitKey(1) & 0xFF == ord('q'):
        break
     i+=1
    
    

            

#cap.release()
cv2.destroyAllWindows()
