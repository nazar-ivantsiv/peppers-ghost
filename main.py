#!bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from random import randint
#from matplotlib import pyplot as plt

win_header = 'view window'
cv2.namedWindow(win_header, cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        #frame = cv2.flip(frame,0)

        img = np.zeros((480, 640, 3), np.uint8)

        p1 = [320, 120]#[randint(120, 200), randint(200, 300)]
        p2 = [160, 360]#[randint(100, 200), randint(200, 300)]
        p3 = [480, 360]#[randint(100, 200), randint(200, 300)]

        pts = np.array([p1,p2,p3], np.int32)
        pts = pts.reshape((-1,1,2))

        frame = cv2.polylines(frame,[pts],True,(0,255,255))

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

'''
############################################################
cap = cv2.VideoCapture('tree.avi')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


############################################################
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap.open()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


##########################################################
win_header = 'view window'
img = cv2.imread('lena.jpg', 0)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])
plt.show()

############################

cv2.namedWindow(win_header, cv2.WINDOW_NORMAL)
cv2.imshow(win_header, img)
k = cv2.waitKey(0) & 0xFF

if k == 27:     # ESC key
    cv2.destroyWindow(win_header)
elif k == ord('s'):
    cv2.imwrite('lena_mod.png',img)
    cv2.destroyAllWindows()
'''
