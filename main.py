#!bin/python
# -*- coding: utf-8 -*-
from __future__ import division

from random import randint
from matplotlib import pyplot as plt
from os import getcwd
import pdb

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
            source: camera (0, 1) or vieofile
        '''
        self.source = source
        self.h = 'SETTINGS'
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            self._cap.open()
        self._out = None
        self.height = int(self._cap.get(4))
        self.width = int(self._cap.get(3))
        self.pyramid_size = self.height * 2
        self.scr_centre = self.pyramid_size / 2
        self.pos = {}

    def run(self):
        '''Starts video processor.'''

        self._init_run()
        fgbg = cv2.createBackgroundSubtractorMOG2(history = 1000,\
                                                    varThreshold = 25,\
                                                    detectShadows = False)
        cap = self._cap      
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                self._get_values()      # Get current positions of trackbars
                if self.pos['BS_on']:   # Apply Background Substraction
                    fgmask = self._fg_mask(frame=frame, fgbg=fgbg, \
                                           k_size=self.pos['k_size'], \
                                           iters=self.pos['iters'],
                                           learn_rate=self.pos['BS_rate'])
                    frame = self._apply_mask(frame, fgmask)
                if (self.pos['i_x'] != 0)or(self.pos['i_y'] != 0):  # Move
                        frame = self._translate(frame, int(self.pos['i_x']), \
                                                int(self.pos['i_y']))
                # Create/Apply triangle mask
                self._triangle_mask(side=self.pos['m_side'], \
                                    centre=self.pos['m_cntr'], \
                                    bottom=self.pos['m_btm'])
                projection = self._apply_mask(frame, self.mask)
                # Create FOUR projections rotated by -90 deg
                self.screen = np.zeros((self.pyramid_size, self.pyramid_size, \
                                        3), np.uint8)
                for i in range(self.pos['projections']):
                    self._add(projection, blend=self.pos['m_blend'])    
                    self.screen = self._rotate(self.screen, -90, \
                                        self.scr_centre, self.scr_centre)
                if self._out != None:
                    self._out.write(self.screen)
                cv2.imshow('screen', self.screen)          # Output SCREEN to window
                if cv2.waitKey(1) & 0xFF == ord('q'):      # Wait for 'q' to exit
                    break
            else:
                break
        self.stop()

    def _init_run(self):
        '''Initializes outputі and adjustments'''
        self.count = 0
        def nothing(x):
            pass
        cv2.namedWindow(self.h, cv2.WINDOW_NORMAL)
        cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("screen", cv2.WND_PROP_FULLSCREEN, \
                              cv2.WINDOW_FULLSCREEN)
        # Create trackbars
        cv2.createTrackbar('mask centre', self.h, int(self.MASK_CENTRE * 100),\
                           100, nothing)
        cv2.createTrackbar('mask bottom', self.h, int(self.MASK_BOTTOM * 100),\
                           100, nothing)
        cv2.createTrackbar('mask side', self.h, 100, 200, nothing)
        cv2.createTrackbar('mask blend', self.h, int(self.MASK_BLEND * 1000),\
                           1000, nothing)
        cv2.createTrackbar('image x', self.h, int(self.width / 2), self.width,\
                             nothing)
        cv2.createTrackbar('image y', self.h, int(self.height / 2), \
                           self.height, nothing)
        cv2.createTrackbar('projections', self.h, 4, 4, nothing)
        #cv2.createTrackbar('image scale', self.h, 10, 19, nothing)

        cv2.createTrackbar('BS on/off\n(wait a second)', self.h, 0, 1, nothing)        
        cv2.createTrackbar('BS learn.rate', self.h, 20, 50, nothing)
        cv2.createTrackbar('dilation kernel size', self.h, 5, 20, nothing)
        cv2.createTrackbar('dilation iters', self.h, 3, 10, nothing)

    def _get_values(self):
        '''Refreshes variables with Trackbars positions (self.pos - var dict)'''

        self.pos['m_cntr'] = cv2.getTrackbarPos('mask centre', self.h) / 100
        self.pos['m_btm'] = cv2.getTrackbarPos('mask bottom', self.h)  / 100
        self.pos['m_side'] = cv2.getTrackbarPos('mask side', self.h) / 100
        self.pos['m_blend'] = cv2.getTrackbarPos('mask blend', self.h) / 1000
        self.pos['i_x'] = cv2.getTrackbarPos('image x', self.h) - self.width / 2
        self.pos['i_y'] = cv2.getTrackbarPos('image y', self.h) - self.height / 2
        self.pos['projections'] = cv2.getTrackbarPos('projections', self.h)
        #i_ratio = cv2.getTrackbarPos('image scale', self.h) / 10 or 1
        self.pos['BS_on'] = cv2.getTrackbarPos('BS on/off\n(wait a second)', \
                                               self.h)
        self.pos['BS_rate'] = cv2.getTrackbarPos('BS learn.rate', self.h) / 10000
        self.pos['k_size'] = cv2.getTrackbarPos('dilation kernel size', \
                                                self.h) or 1
        self.pos['iters'] = cv2.getTrackbarPos('dilation iters', self.h)

    def set_output(self, file_path):
        '''Define the codec and create VideoWriter object to output video.
        Args:
            file_path: filename or path to output file
        '''
        print('Writing video to: {}'.format(file_path))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._out = cv2.VideoWriter(file_path, fourcc, 20.0, \
                            (self.pyramid_size, self.pyramid_size))

    def stop(self):
        '''Release everything if job is finished. And close the window.
        '''
        self._cap.release()
        if self._out != None:
            self._out.release()
        cv2.destroyAllWindows()

    def _triangle_mask(self, side=1, centre=MASK_CENTRE, \
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
        left_pt = [ (centre_x - (width / 2) * side), \
                    (centre_y - height / 2)]
        right_pt = [(centre_x + (width / 2 * side)), \
                    (centre_y - height / 2)]
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

    def _fg_mask(self, frame, fgbg, k_size, iters, learn_rate):
        '''Background substraction MOG2 algorithm.
        Creates FOREGROUND mask
        Args:
            frame: 
            fgbg: MOG2 instance
            k_size: kernel size (for morphologyEx operations)
            iters: number of iterations for Dilation
            learn_rate: learning rate for MOG2 alg (def 0.002)
        Returns:
            fgmask: FOREGROUND mask
        '''
        # Get FGMASK with MOG2
        fgmask = fgbg.apply(frame, learningRate=learn_rate)
        print(self.count)
        self.count += 1
        if self.count % 100 == 0:
            cv2.imwrite('fgmask{}.png'.format(self.count), fgmask)
        # Elliptical Kernel for morphology func
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k_size, k_size))
        # Dilation alg (increases white regions size)
        fgmask = cv2.dilate(fgmask, kernel, iterations = iters)        
        # Closing (remove black points from the object)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        # Draw white border
        cv2.rectangle(fgmask, (0, 0), (self.width, self.height), \
                      (255, 255, 255), 3)

        # Fill Holes (flood fill the object)
        #fgmask = self._imfill_holes(cv2.bitwise_not(fgmask))       
        return fgmask

    @staticmethod
    def _apply_mask(img, mask):
        '''Apply mask to image
        '''
        return cv2.bitwise_and(img, img, mask=mask)

    def _add(self, projection, blend=MASK_BLEND):
        '''Adds PROJECTION to bottom centre of SCREEN'''

        p_height, p_width = projection.shape[:2]
        #Extract ROI
        x = self.scr_centre - p_width / 2    
        y = self.scr_centre - p_height * blend
        roi = self.screen[y : y + p_height, x : x + p_width]
        # Black-out the area of projection in ROI
        roi_bg = self._apply_mask(roi, self.mask_inv)
        # Add WINDOW to ROI
        dst = cv2.add(roi_bg, projection)
        # Apply ROI to SCREEN
        self.screen[y : y + p_height, x : x + p_width] = dst

    @staticmethod
    def _imfill_holes(im_in):
        '''Floodfill white holes in binary image'''

        th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV);
        # Copy the thresholded image.
        im_floodfill = im_th.copy()         
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 255);
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        # Combine the two images to get the foreground.
        im_out = im_th | im_floodfill_inv
        return im_out


    def _scale(self, img, ratio):
        '''Uniform scale
        '''
        return cv2.resize(img, None, fx=ratio, fy=ratio)

    def _show(self, img):
        cv2.imshow('show', img)

        k = cv2.waitKey(0) & 0xFF

        if k == 27:     # ESC key
            cv2.destroyWindow('show')

    @staticmethod
    def _translate(img, x_dist, y_dist):
        rows, cols = img.shape[:2]
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
    def _show_plt(img):
        plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
        #plt.xticks([]), plt.yticks([])
        plt.show()


#ghost = Ghost('/home/chip/pythoncourse/hologram2/test.mp4')
ghost = Ghost()

#path = getcwd() + '/out.avi'
#ghost.set_output(path)

ghost.run()


