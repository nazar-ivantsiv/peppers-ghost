import cv2
import numpy as np
from matplotlib import pyplot as plt

win_header = 'view window'
# Create a black image
img = np.zeros((512,512,3), np.uint8)


pts = np.array([[200,300],[370,400],[500,100]], np.int32)
pts = pts.reshape((-1,1,2))

img = cv2.polylines(img,[pts],True,(0,255,255))

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'Chip',(10,300), font, 4,(255,255,255),2,cv2.LINE_AA)

cv2.namedWindow(win_header, cv2.WINDOW_NORMAL)
cv2.imshow(win_header, img)
k = cv2.waitKey(0) & 0xFF

if k == 27:     # ESC key
    cv2.destroyWindow(win_header)

'''
# Draw a diagonal blue line with thickness of 5 px
img = cv2.line(img,(0,0),(511,511),(255,0,0),5)

img = cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

img = cv2.circle(img,(447,63), 63, (0,0,255), -1)

img = cv2.ellipse(img,(256,256),(100,50),0,0,360,255,-1)

'''