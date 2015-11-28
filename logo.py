import cv2
import numpy as np

def show(img):
    cv2.imshow(win_header, img)

    k = cv2.waitKey(0) & 0xFF

    if k == 27:     # ESC key
        cv2.destroyWindow(win_header)

win_header = 'view window'
cv2.namedWindow(win_header, cv2.WINDOW_NORMAL)

# Load two images
img1 = cv2.imread('me.jpg')
rows, cols, channel = img1.shape

# I want to put logo on top-left corner, So I create a ROI

# Now create a mask of logo and create its inverse mask also
# Create a black image
img = np.zeros((rows, cols, 3), np.uint8)

#p1 = [320, 120]#[randint(120, 200), randint(200, 300)]
#p2 = [160, 360]#[randint(100, 200), randint(200, 300)]
#p3 = [480, 360]#[randint(100, 200), randint(200, 300)]

p1 = [cols/2, 0]
p2 = [0, rows]
p3 = [cols, rows]

pts = np.array([p1,p2,p3], np.int32)
#pts = pts.reshape((-1,1,2))

mask = cv2.fillConvexPoly(img, pts, (255,255,255), 1)
#mask_inv = cv2.bitwise_not(mask)
#show(mask)



mask2gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(mask2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
#show(mask_inv)

print(img1.dtype)
print(mask_inv.dtype)

# Now black-out the area of logo in ROI
#img1_bg = cv2.bitwise_and(img1,img1,mask = mask)
#show(img_bg)

# Take only region of logo from logo image.
img1_fg = cv2.bitwise_and(img1, img1, mask=mask)
#show(img1_fg)

#show(img2_fg)
# Put logo in ROI and modify the main image
#dst = cv2.add(img1,mask)

show(img1_fg)