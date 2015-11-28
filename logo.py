import cv2
import numpy as np


win_header = 'view window'
# Create a black image
img = np.zeros((512,512,3), np.uint8)

img = cv2.ellipse(img,(256,256),(100,50),0,0,360,255,-1)


font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'Chip',(10,300), font, 4,(255,255,255),2,cv2.LINE_AA)

def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),100,(255,0,0),-1)


cv2.setMouseCallback(win_header,draw_circle)

cv2.namedWindow(win_header, cv2.WINDOW_NORMAL)

cv2.imshow(win_header, img)
k = cv2.waitKey(0) & 0xFF

if k == 27:     # ESC key
    cv2.destroyWindow(win_header)