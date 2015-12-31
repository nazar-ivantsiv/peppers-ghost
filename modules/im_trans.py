from __future__ import division
from . import cv2
from . import np


def apply_mask(img, mask):
    """Apply bitwise mask to image"""
    return cv2.bitwise_and(img, img, mask=mask)

def brightness_contrast(img, alpha=1, beta=0):
    """Adjust brightness and contrast.
    Args:
        alpha -- contrast coefficient (1.0 - 3.0)
        beta -- brightness increment (0 - 100)
    Returns:
        result -- image with adjustments applied
    """
    result = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for i in range(3):
        result[:, :, i] = cv2.add(cv2.multiply(img[:, :, i], alpha), beta)
    return result

def create_triangle_mask(height, width, side=1.5, centre=0, bottom=1):
    """Creates triangle mask. According to proportions of the frame(img).
    Args:
        height -- original frame(img) height
        width -- original frame(img) width
        side -- scale factor of hypotenuse of triangle mask
        centre -- centre of the mask in frame(img)
        bottom -- scale factor of legs of triangle mask
    Returns:
        result -- triangle mask
    """
    # Set mask points
    centre_y = height / 2 + height / 2 * centre
    centre_x = width / 2 
    left_pt = [(centre_x - (width / 2) * side), \
                (centre_y - height / 2)]
    right_pt = [(centre_x + (width / 2 * side)), \
                (centre_y - height / 2)]
    bottom_pt = [centre_x, centre_y + (height / 2) * bottom]
    pts = np.array([bottom_pt, left_pt, right_pt], np.int32)
    # Black image
    result = np.zeros((height, width, 3), np.uint8)
    # Create traiangle
    result = cv2.fillConvexPoly(result, pts, (255, 255, 255), 1)
    # Convert to GRAY
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    return result

def draw_rect(img, faces):
    """Draw BLUE rectangle on img.
    Args:
        img -- image
        faces -- face coords (x, y, w, h)
    Returns:
        img -- image with BLUE rectangle around the face
    """
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img

def draw_ellipse(img, faces):
    """Draws solid filled WHITE ellipse fitted into rectangle area.
    Args:
        img -- image (frame)
        faces -- ist of np.array([x,y,w,h]) face coordinates
    Returns:
        result -- img with white elipses
    """
    for (x, y, w, h) in faces:
        result = cv2.ellipse(img, (x + w // 2, y + h // 2), \
                                (w // 2, int(h / 1.5)), \
                                0, 0, 360, (255, 255, 255), -1)
    return result

def fit_into(img, dest_height, dest_width):
    """Fits img into rect (height x width) or crops it.
    Args:
        img -- image
        h -- destination height
        w -- destination width
    Returns:
        result -- img cropped with (dest_height X dest_width) if img is BIGGER
        than destination dimentions. img with border added if it is SMALLER, to
        fit into destination dimentions.
    """
    result = img.copy()
    im_height, im_width = img.shape[:2]
    # Get x, y coordinates for cropped subframe
    y = (im_height - dest_height) // 2
    x = (im_width - dest_width) // 2
    # Border color
    BLACK = [0, 0, 0]
    if y < 0:
        # Add border to TOP and BOTTOM of img
        # Will compensate remaining (y) lack of height
        result = cv2.copyMakeBorder(result, abs(y), abs(y), 0, 0, \
                                    cv2.BORDER_CONSTANT, \
                                    value=BLACK)
        y = 0
    if x < 0:
        # Add border to LEFT and RIGHT of img.
        # Will compensate remaining (x) lack of width
        result = cv2.copyMakeBorder(result, 0, 0, abs(x), abs(x), \
                                    cv2.BORDER_CONSTANT, \
                                    value=BLACK)
        x = 0
    return result[y:y + dest_height, x:x + dest_width]

def img_to_gray(img):
    """Converts img (frame) to Grayscale, adjust contrast."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(img) # Adj. contrast

def scale(img, ratio):
    """Uniform scale.
    Args:
        img -- image
        ratio -- scale ratio (1 == 100%)
    Returns:
        result -- scaled image
    """
    return cv2.resize(img, None, fx=ratio, fy=ratio)

def show(img):
    """Shows img in the 'show' window"""
    cv2.imshow('show', img)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:     # ESC key
        cv2.destroyWindow('show')

def rotate(img, angle, anchor_x, anchor_y):
    """Rotates image about anchor point
    Args:
        img -- image
        angel -- angle to rotate
        anchor_x, anchor_y -- anchor coordinates to rotate about.
    Returns:
        result -- rotated image (no scaling)
    """
    height, width = img.shape[:2]
    M = cv2.getRotationMatrix2D((anchor_x, anchor_y), angle, 1)
    result = cv2.warpAffine(img, M, (width, height))
    return result    

def translate(img, x_dest, y_dest):
    """Translate img to x, y coordinates.
    Args:
        img -- image
        x_dest -- x coordinat
        y_dest -- y coord
    Returns:
        result -- translated img
    """
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, x_dest], [0, 1, y_dest]])
    result = cv2.warpAffine(img, M, (cols, rows))
    return result