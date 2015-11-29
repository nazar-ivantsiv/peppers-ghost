#!bin/python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import cv2
from random import randint
from matplotlib import pyplot as plt

win_header = 'view window'
cv2.namedWindow(win_header, cv2.WINDOW_NORMAL)

def show(img):
    #print(img.shape)
    cv2.imshow(win_header, img)

    k = cv2.waitKey(0) & 0xFF

    if k == 27:     # ESC key
        cv2.destroyWindow(win_header)

def show_plt(img):
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    #plt.xticks([]), plt.yticks([])
    plt.show()

def create_masks(height, width):
    '''Creates two mask shapes
    Args:
        height: original image HEIGHT
        width: original image WIDTH
    Return:
        mask_top_bottom: top and bottom rectangles mask
        mask_left_right: left and right rectangles mask
    '''
    # Scaled up 200%
    height_2x = height
    width_2x = width

    # Set TOP_BOTTOM mask points
    top_pt = [0.5 * width_2x, 0]
    left_pt = [- (0.5 * width_2x), height_2x]
    right_pt = [width_2x*1.5, height_2x]
    top_bottom_mask = mask(height_2x, width_2x, top_pt, left_pt, right_pt)
    #show(top_bottom_mask)
    # Set TOP_BOTTOM mask points
    top_pt = [0.5 * width_2x, 0]
    left_pt = [width_2x, -1/3 * height_2x]
    right_pt = [width_2x, 5/3 * height_2x]
    left_right_mask = mask(height_2x, width_2x, top_pt, left_pt, right_pt)
    #show(left_right_mask)
    return top_bottom_mask, left_right_mask

def mask(height_2x, width_2x, top_pt, left_pt, right_pt):
    '''Creates triangle mask from 3x points
    Args:
        top_pt
        left_pt
        right_pt
    Return:
        mask
    '''
    print(top_pt, left_pt, right_pt)
    mask = np.zeros((height_2x, width_2x, 3), np.uint8)
    show_plt(mask)
    pts = np.array([top_pt, left_pt, right_pt], np.int32)
    #pts = pts.reshape((-1,1,2))
    mask = cv2.fillConvexPoly(mask, pts, (255,255,255), 1)
    show_plt(mask)
    mask2grey = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(mask2grey, 10, 255, cv2.THRESH_BINARY)
    show(mask)
    return mask

def crop_triangles(frame, mask, height, width):
    '''Crop triangels by mask
    '''
    # Apply MASK. ???????

    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Scale down 
    #print(height, width)
    result = cv2.resize(result, (height//2, width//2), interpolation = cv2.INTER_CUBIC)

    # Rotate image 90 CCW
    #M = cv2.getRotationMatrix2D((width/2,height/2),90,1)
    #result = cv2.warpAffine(result,M,(width,height))

    return result


cap = cv2.VideoCapture(0)
# Get image dimensions ( HIGH x WIDTH )
# 480 x 640
height = int(cap.get(4))
width = int(cap.get(3))
# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

# CREATE MASKs 
top_bottom_mask, left_right_mask = create_masks(height, width)


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
 
        top_bottom = crop_triangles(frame, top_bottom_mask, height, width)
        left_right = crop_triangles(frame, left_right_mask, height, width)

        #frame = cv2.flip(frame,0)

        # write the frame
        #out.write(frame)

        #cv2.imshow(win_header,left_right)
        #cv2.imshow(win_header,frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
#out.release()
cv2.destroyAllWindows()