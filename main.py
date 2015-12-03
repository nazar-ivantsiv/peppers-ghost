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


class Ghost(object):
    '''Pepper's Ghost video processor. 
    Creates pyramid like projection of input video (camera or file), ready to
    project on pseudo holographic pyramid.
    Args:
        source: video camera or file
    Returns:

    '''
    def __init__(self, source=0):
        '''Args:
            source: camera (0, 1) or vieofile (file.avi)
        '''
        self.source = source
        self.win_header = 'Peper\'s Ghost'
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            self._cap.open()

        self._out = None
        self.height = int(self._cap.get(4))
        self.width = int(self._cap.get(3))

        self.pyramid_size = self.height * 2

        self.screen = np.zeros((self.pyramid_size, self.pyramid_size, 3), \
                                  np.uint8)
        self.scr_centre = self.pyramid_size /2

        self.mask = self._create_mask(centre=0.5)

        self.mask_inv = cv2.bitwise_not(self.mask)

        #self._show_plt(self.mask)


    def run(self):
        '''Starts video processor.
        '''
        cv2.namedWindow(self.win_header, cv2.WINDOW_NORMAL)
        # Remember to check if the output is initialized.

        cap = self._cap
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                # PUT MASK ASSIGNMENT HERE TO HAVE INTERACTION

                # Create BOTTOM projection (flip + apply mask)
                projection = self._apply_mask(frame, self.mask)

                # Create empty SCREEN. !!! BAD !!! Move OUT of cycle???
                self.screen = np.zeros((self.pyramid_size, self.pyramid_size,\
                                         3), np.uint8)

                #print(screen.shape)
                #print(projection.shape)

                for i in range(4):
                    self._add(projection)    
                    self.screen = self._rotate(self.screen, -90, \
                                           self.scr_centre, self.scr_centre)

                if self._out != None:
                    self._out.write(self.screen)

                #cv2.imshow(win_header,left_right)
                cv2.imshow(self.win_header, self.screen)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break


        self.stop()

    def set_output(file_path):
        '''Define the codec and create VideoWriter object to output video.
        Args:
            file_path: filename or path to output file
        '''
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._out = cv2.VideoWriter(file_path, fourcc, 20.0, \
                                   (height * 2, height * 2))

    def stop(self):
        '''Release everything if job is finished. And close the window.
        '''
        self._cap.release()
        if self._out != None:
            self._out.release()
        cv2.destroyAllWindows()

    def _create_mask(self, scale_x=1, scale_y=1, centre=2, sharp=0.33):
        '''Creates mask.
        Args:
            height: original image HEIGHT
            width: original image WIDTH
            scale_x: scale mask height
            scale_y: scale mask width
            centre: move mask centre by Y axis
        Return:
            mask: triangle mask
        '''
        # Set mask points
        height = self.height
        width = self.width
        
        centre_y = height / 2 + height / 2 * centre
        centre_x = width / 2 

        left_pt = [ (centre_x - width / 2) * scale_x, \
                    (centre_y - height / 2) * scale_y]

        right_pt = [(centre_x + width / 2) * scale_x, \
                    (centre_y - height / 2) * scale_y]

        bottom_pt = [centre_x, centre_y + (height / 2) * sharp]

        pts = np.array([bottom_pt, left_pt, right_pt], np.int32)
        # Black image
        result = np.zeros((self.height, self.width, 3), np.uint8)
        #pts = pts.reshape((-1,1,2))
        # Create traiangle
        result = cv2.fillConvexPoly(result, pts, (255,255,255), 1)
        # Convert to GRAY
        result = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
        #ret, result = cv2.threshold(mask2grey, 10, 255, cv2.THRESH_BINARY)
        return result

    def _add(self, window, blend=0.915):

        w_height, w_width = window.shape[:2]

        # CENTRE of SCREEN (ROI coordinates)
        centre = self.pyramid_size / 2

        y = centre - w_width / 2    
        x = centre - w_height * blend

        # Create ROI for (y : y + b_height, x : x + b_width) region
        roi = self.screen[x : x + w_height, y : y + w_width]
        #### Apply mask & mask_inverse to ROI ###
        # Black-out the area of window in ROI
        roi_bg = self._apply_mask(roi, self.mask_inv)

        # Add WINDOW to ROI
        #dst = cv2.add(cv2.bitwise_not(roi_bg), window)
        dst = cv2.add(roi_bg, window)
        # Apply ROI to SCREEN
        self.screen[x : x + w_height, y : y + w_width] = dst
        #self._show_plt(self.screen)

    @staticmethod
    def _show(img):
        cv2.imshow(win_header, img)

        k = cv2.waitKey(0) & 0xFF

        if k == 27:     # ESC key
            cv2.destroyWindow(win_header)

    @staticmethod
    def _show_plt(img):
        plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
        #plt.xticks([]), plt.yticks([])
        plt.show()

    @staticmethod
    def _scale(img, ratio):
        '''Uniform scale
        '''
        return cv2.resize(img,None,fx=ratio, fy=ratio, \
                          interpolation = cv2.INTER_CUBIC)

    @staticmethod
    def _rotate(img, angle, anchor_x, anchor_y):
        '''Rotates image about anchor point
        Args:
            img: image
            angel: angle to rotate
            anchor_x, anchor_y: anchor coordinates to rotate about.
        Returns:
            rotated image (no scaling)
        '''
        height, width = img.shape[:2]
        M = cv2.getRotationMatrix2D((anchor_x, anchor_y), angle, 1)
        result = cv2.warpAffine(img, M, (width, height))
        return result    

    @staticmethod
    def _apply_mask(img, mask):
        '''Applies mask to image
        '''
        return cv2.bitwise_and(img, img, mask=mask)

    @staticmethod
    def _rotate_pt(x ,y, angle_degree, anchor_x=0, anchor_y=0):
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


ghost = Ghost()
ghost.run()