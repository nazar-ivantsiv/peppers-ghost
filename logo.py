import cv2
import numpy as np
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

def scale(img, ratio):
    return cv2.resize(img,None,fx=ratio, fy=ratio, interpolation = cv2.INTER_CUBIC)

#win_header = 'view window'
#cv2.namedWindow(win_header, cv2.WINDOW_NORMAL)

img = cv2.imread('lena.jpg')
height, width, channel = img.shape

show_plt(scale(img, 0.5))