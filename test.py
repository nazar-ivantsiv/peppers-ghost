#!bin/python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import cv2

from math import cos
from math import sin
from math import pi
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

def create_mask(height, width):
    '''Creates mask for BOTTOM element
    Args:
        height: original image HEIGHT
        width: original image WIDTH
    Return:
        mask: BOTTOM mask
    '''
    # Set BOTTOM mask points
    top_pt = [width / 2, 0]
    left_pt = [-1/2 * width, height]
    right_pt = [3/2 * width, height]
    pts = np.array([top_pt, left_pt, right_pt], np.int32)
    # Black image
    mask = np.zeros((height, width, 3), np.uint8)
    #pts = pts.reshape((-1,1,2))
    # Create traiangle
    mask = cv2.fillConvexPoly(mask, pts, (255,255,255), 1)
    # Convert to GRAY
    mask2grey = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    ret, result = cv2.threshold(mask2grey, 10, 255, cv2.THRESH_BINARY)
    return cv2.flip(result, 0)

def crop_triangle(img, mask):
    '''Crop triangel by mask
    '''
    return cv2.bitwise_and(img, img, mask=mask)

def scale(img, ratio):
    '''Uniform scale
    '''
    return cv2.resize(img,None,fx=ratio, fy=ratio, \
                      interpolation = cv2.INTER_CUBIC)

def rotate(img, angle, anchor_x, anchor_y):
    '''Rotates image about anchor point
    '''
    height, width = img.shape[:2]

    M = cv2.getRotationMatrix2D((anchor_x, anchor_y), angle, 1)
    print(M)
    result = cv2.warpAffine(img, M, (width, height))
    return result

def rotate_pt(x ,y, angle_degree, anchor_x=0, anchor_y=0):
    '''Rotate point by anchor
    Agrs: 
        x: coordinate
        y: coordinate
        angle: in DEGREES
    Returns:
        (x', y'): rotated coordinates
    '''
    angle = angle_degree * pi / 180    # Convert from degrees to radians
    x2 = ((x - anchor_x) * cos(angle)) - ((y - anchor_y) * sin(angle)) + anchor_x
    y2 = ((x - anchor_x) * sin(angle)) + ((y - anchor_y) * cos(angle)) + anchor_y

    return x2, y2

def add(img1, img2):
    pass

win_header = 'view window'
cv2.namedWindow(win_header, cv2.WINDOW_NORMAL)

#img = cv2.imread('aero3.jpg')
img = cv2.imread('lena.jpg')
# Get image dimensions ( HIGH x WIDTH )
# 480 x 640
height, width, channels = img.shape

mask = create_mask(height, width)

top = crop_triangle(img, mask)
bottom = cv2.flip(top, 0)

# 2 * Height - length of side of new RECTANGLE
screen = np.zeros((height * 2, width * 2, 3), np.uint8)
#show_plt(screen)

print(screen.shape)
print(bottom.shape)

scr_height, scr_width = screen.shape[:2]
b_height, b_width = bottom.shape[:2]

# TOP LH Corner of BOTTOM image on SCREEN (ROI coordinates
x = scr_width / 4
y = scr_height / 2

# Create ROI for (y : y + b_height, x : x + b_width) region
# Apply mask & mask_inverse to ROI
# ADD

# Apply ROI to SCREEN
screen[y : y + b_height, x : x + b_width] = bottom



screen = rotate(screen, -90, scr_height / 2, scr_width / 2)

show_plt(screen)

#if cv2.waitKey(1) & 0xFF == ord('q'):
#    cv2.destroyAllWindows()