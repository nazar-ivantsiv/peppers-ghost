#!bin/python
# -*- coding: utf-8 -*-
from __future__ import division

from random import randint
from matplotlib import pyplot as plt

import numpy as np
import cv2


class Ghost(object):
    '''Pepper's Ghost video processor. 
    Creates pyramid like projection of input video (camera or file), ready to
    project on pseudo holographic pyramid.
    Args:
        source: video camera or file
    '''
    MASK_CENTRE = 0.5
    MASK_BOTTOM = 0.33
    MASK_BLEND = 0.915

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
        self._create_mask()

        #self._show_plt(self.mask)

    def init(self):
        '''Initialize main window'''

    def run(self):
        '''Starts video processor.
        '''
        cv2.namedWindow(self.win_header, cv2.WINDOW_NORMAL)
        fgbg = cv2.createBackgroundSubtractorMOG2(history = 500, \
                                                  varThreshold = 16, \
                                                  detectShadows = False)

        cap = self._cap
        #raw_input('Nobody in fronet of camera, only BACKGROUND?')
        #background = cap.read()[1]
        #self._show(background)

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                # PUT MASK ASSIGNMENT HERE TO HAVE INTERACTION

                #frame_inv = cv2.bitwise_not(frame)

                fgmask = fgbg.apply(frame)

                frame = self._apply_mask(frame, fgmask)

                projection = self._apply_mask(frame, self.mask)


                #self._show(fgmask)
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

    @property
    def get_frame(self):
        return self._cap.read()
    
    def _create_mask(self, scale_x=1, scale_y=1, centre=MASK_CENTRE, \
                     bottom=MASK_BOTTOM):
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
        height = self.height
        width = self.width
        # Set mask points
        centre_y = height / 2 + height / 2 * centre
        centre_x = width / 2 
        left_pt = [ (centre_x - width / 2) * scale_x, \
                    (centre_y - height / 2) * scale_y]
        right_pt = [(centre_x + width / 2) * scale_x, \
                    (centre_y - height / 2) * scale_y]
        bottom_pt = [centre_x, centre_y + (height / 2) * bottom]
        pts = np.array([bottom_pt, left_pt, right_pt], np.int32)
        # Black image
        result = np.zeros((self.height, self.width, 3), np.uint8)
        # Create traiangle
        result = cv2.fillConvexPoly(result, pts, (255,255,255), 1)
        # Convert to GRAY
        result = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
        self.mask = result
        self.mask_inv = cv2.bitwise_not(self.mask)

    def _add(self, window, blend=MASK_BLEND):

        w_height, w_width = window.shape[:2]

        # CENTRE of SCREEN (ROI coordinates)
        centre = self.pyramid_size / 2

        x = centre - w_width / 2    
        y = centre - w_height * blend

        # Create ROI for (y : y + b_height, x : x + b_width) region
        roi = self.screen[y : y + w_height, x : x + w_width]
        #### Apply mask & mask_inverse to ROI ###
        # Black-out the area of window in ROI
        roi_bg = self._apply_mask(roi, self.mask_inv)

        # Add WINDOW to ROI
        #dst = cv2.add(cv2.bitwise_not(roi_bg), window)
        dst = cv2.add(roi_bg, window)
        # Apply ROI to SCREEN
        self.screen[y : y + w_height, x : x + w_width] = dst
        #self._show_plt(self.screen)

    def _show(self, img):
        cv2.imshow(self.win_header, img)

        k = cv2.waitKey(0) & 0xFF

        if k == 27:     # ESC key
            cv2.destroyWindow(self.win_header)

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
    def _translate(img, x_dist, y_dist):
        M = np.float32([[1, 0, x_dist],[0, 1, y_dist]])
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst

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

ghost = Ghost()
ghost.run()
