#!bin/python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import cv2
from random import randint
from matplotlib import pyplot as plt


def show(img):
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
    # Set TOP_BOTTOM mask points
    top_pt = [width / 2, 0]
    left_pt = [0, height / 2]
    right_pt = [width, height / 2]
    pts = np.array([top_pt, left_pt, right_pt], np.int32)
    top_bottom_mask = mask(height, width, pts)

    # Set TOP_BOTTOM mask points
    top_pt = [width / 2, 0]
    left_pt = [(width - height) / 2, width / 2]
    right_pt = [(width - height) / 2 + height, width / 2]
    pts = np.array([top_pt, left_pt, right_pt], np.int32)
    left_right_mask = mask(height, width, pts)

    return top_bottom_mask, left_right_mask

def mask(height, width, pts):
    '''Creates triangle mask from 3x points
    Args:
        top_pt
        left_pt
        right_pt
    Return:
        mask
    '''
    print(pts)
    mask = np.zeros((height, width, 3), np.uint8)
    #pts = pts.reshape((-1,1,2))
    mask = cv2.fillConvexPoly(mask, pts, (255,255,255), 1)
    
    mask2grey = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(mask2grey, 10, 255, cv2.THRESH_BINARY)
    #show(mask)
    return mask

def crop_triangles(img, mask):
    '''Crop triangels by mask
    '''
    return cv2.bitwise_and(img, img, mask=mask)

def scale(img, scalar):
    height, width = img.shape[:2]
    result = cv2.resize(img, (int(width/scalar), int(height/scalar)), \
                        interpolation = cv2.INTER_CUBIC)
    return result

def rotate(img, angle):
    '''Rotates image about its centre
    '''
    height, width = img.shape[:2]
    M = cv2.getRotationMatrix2D((width, height), angle, 1)
    result = cv2.warpAffine(result, M, (width, height))
    return result


win_header = 'view window'
cv2.namedWindow(win_header, cv2.WINDOW_NORMAL)

img = cv2.imread('aero3.jpg', 0)
#img = cv2.imread('lena.jpg', 0)
# Get image dimensions ( HIGH x WIDTH )
# 480 x 640
height, width = img.shape

# CREATE MASKs 
top_bottom_mask, left_right_mask = create_masks(height, width)

 
top_bottom = crop_triangles(img, top_bottom_mask)
left_right = crop_triangles(img, left_right_mask)

show(img)
show(top_bottom)
show(left_right)

if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.destroyAllWindows()