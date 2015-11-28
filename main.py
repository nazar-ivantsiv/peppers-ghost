#!bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from random import randint
#from matplotlib import pyplot as plt

win_header = 'view window'
cv2.namedWindow(win_header, cv2.WINDOW_NORMAL)

def process(frame, mask, rows, cols):

    # Apply MASK.
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Scale 

    # Rotate image 90 CCW
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    result = cv2.warpAffine(result,M,(cols,rows))

    return result


cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:

        rows, cols = frame.shape[:2]
        print(rows, cols)

        ###### CREATE MASK ######
        # Create a black image
        img = np.zeros((rows, cols, 3), np.uint8)

        p1 = [cols/2, 0]
        p2 = [0, rows]
        p3 = [cols, rows]
        pts = np.array([p1,p2,p3], np.int32)
        #pts = pts.reshape((-1,1,2))

        mask = cv2.fillConvexPoly(img, pts, (255,255,255), 1)
        #show(mask)
        mask2gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        
        ret, mask = cv2.threshold(mask2gray, 10, 255, cv2.THRESH_BINARY)

        frame = process(frame, mask, rows, cols)

        #frame = cv2.flip(frame,0)

        # write the frame
        out.write(frame)

        cv2.imshow(win_header,frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()