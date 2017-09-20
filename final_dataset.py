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


pinkLower = (90, 30, 150)
pinkUpper = (200, 100, 255)
blueLower = (100, 30, 3)
blueUpper = (255, 140, 80)
greenLower = (0, 100, 0)
greenUpper = (60, 225, 30)


from math import acos
from math import sqrt
from math import pi

def length(v):
    return sqrt(v[0]**2+v[1]**2)
def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]
def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]
def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=acos(cosx) # in radians
   return rad*180/pi # returns degrees
def angle_clockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
         inner=inner-90
         if inner<0:
              inner=360+inner
         return inner
    else: # if the det > 0 then A is immediately clockwise of B
         inner=360-inner-90
         if inner<0:
              inner=inner-360
         return inner
def angle_loc(bx,by,px,py,gx,gy):
    

     a = np.array([bx,by])
     b = np.array([gx,gy])
     c = np.array([px,py])

     ba = a - b
     bc = c - b

##     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
##     angle = np.arccos(cosine_angle)
     print angle_clockwise(ba,bc)
     return angle_clockwise(ba,bc)
def locate(frame,up,lw):
     x=0
     y=0
     frame = imutils.resize(frame, width=600)
     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     #frame =colorext(frame)
     #mask = cv2.inRange(frame,  up,lw)
     mask = cv2.inRange(frame, lw, up)
     mask = cv2.erode(mask, None, iterations=6)
     mask = cv2.dilate(mask, None, iterations=2)
     #cv2.imshow('mask',mask)

     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		             cv2.CHAIN_APPROX_SIMPLE)[-2]
     center = None
   
     if len(cnts) > 0:
                
             c = max(cnts, key=cv2.contourArea)
             ((x, y), radius) = cv2.minEnclosingCircle(c)
                #print x, y
             M = cv2.moments(c)
             center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
             if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
                    cv2.circle(hsv, (int(x), int(y)), int(radius),
			(0, 255, 255), 2)
                    cv2.circle(hsv, center, 5, (0, 0, 255), -1)
     #cv2.imshow('frame',hsv)
     return  x,y,hsv



train='/home/nsr/Documents/dataset_car'
for img in tqdm(sorted(os.listdir(train))):
    time.sleep(0.1)
    path =os.path.join(train,img)
    img=cv2.imread(path)
    px,py,hsv=locate(img,pinkUpper,pinkLower)
    bx,by,hsv=locate(img,blueUpper,blueLower)
    gx,gy,hsv=locate(img,greenUpper,greenLower)
    print bx,by,px,py,gx,gy
    angle=angle_loc(bx,by,px,py,gx,gy)
    
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
