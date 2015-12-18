#!bin/python
'''
==============================================================================
Pepper's Ghost video processor.

Author:
    Nazar Ivantsiv
Mentors:
    Nazar Grabovskyy
    Igor Lushchik

Creates pyramid like projection of input video (camera or file), ready to
use with pseudo holographic pyramid.
Source - video camera or video file

USAGE:
    Ghost class creates two main windows (SETTINGS and screen). 'screen' will 
open in the desktop that you are executing program at. Runs in FULLSCREEN 
mode.
    'SETTINGS' window gives you ability to change different Ghost instance
attributes. In the bottom is a 'debugger' trackbar to turn on debugger mode.

Key 'q' - To quit
Key 'd' - Debugger windows (on/off)

==============================================================================
'''

# -*- coding: utf-8 -*-
from __future__ import division
from os import getcwd

import numpy as np
import cv2

# Features for CascadeClassifier (frontal face)
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

class Ghost(object):
    '''Pepper's Ghost video processor.'''
    DEBUGGER_MODE = 0                           # Flag: debugger mode off
    MASK_CENTRE = 0                             # Centre of the mask in SCREEN
    MASK_BOTTOM = 1                             # Mask bottom corner position
    MASK_SIDE = 1.5                             # Mask side corners pos
    MASK_BLEND = 1                              # Centre of the mask in FRAME


    def __init__(self, source=0):
        '''Args:
            source -- camera (0, 1) or vieofile
        '''
        self.source = source                    # Input source num/path
        self.h = 'SETTINGS'
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            self._cap.open()                    # Input source instance
        self._out = None                        # Output source instance(file)
        self.loop_video = 1                     # Flag: loop video
        self.height = int(self._cap.get(4))     # Frame height
        self.width = int(self._cap.get(3))      # Frame width
        self.pyramid_size = self.height * 2     # Size of pyramide side
        self.scr_centre_x = self.width          # Scr centre y
        self.scr_centre_y = self.height         # Scr centre x
        self.pos = {}                           # Trackbar positions
            # Faces detected (by default - rect in the centre)
        x = self.width // 2                     # Frame centre x
        y = self.height // 2                    # Frame centre y
        a = x // 2
        self.faces = [np.array([x - a, y - a, x, y])]
        self.gc_rect = (x - a, y - a, x, y)     # GrabCut default rect
        self.def_face = self.faces              # Default value (rect in centre)
        # Cascade clsfr GC instanece
        self._face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        self._debugger_off = not self.DEBUGGER_MODE # Flag: debugger status

    def run(self):
        '''Video processing.'''
        self._init_run()
        cap = self._cap
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:

                # Create projection with all available masks applied to frame
                projection = self._apply_settings(frame)                   
                self._rotate_projection(projection)
                if self._out != None:               # Save video to file
                    self._out.write(self.screen)
                cv2.imshow('screen', self.screen)   # Output SCREEN to window
                self._loop_video()                  # Loop the video

                key_pressed = 0xFF & cv2.waitKey(1)
                if key_pressed == ord('q'):         # Wait for 'q' to exit
                    break
                elif key_pressed == ord('d'):       # Debugger windows(on/off)
                    self.DEBUGGER_MODE = not self.DEBUGGER_MODE

                self._debugger_mode(frame, projection)
            else:
                break
        self.stop()

    def set_output(self, file_path):
        '''Define the codec and create VideoWriter object to output video.
        Args:
            file_path: filename or path to output file
        '''
        print('Writing video to: {}'.format(file_path))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._out = cv2.VideoWriter(file_path, fourcc, 20.0, \
                            (self.width * 2, self.height * 2))
        print(self.width * 2, self.height * 2)

    def stop(self):
        '''Release everything if job is finished. And close the windows.'''
        self._cap.release()
        if self._out != None:
            self._out.release()
        cv2.destroyAllWindows()

    @staticmethod
    def _brightness_contrast(img, alpha, beta):
        '''Adjust brightness and contrast.
        Args:
            alpha -- contrast coefficient (1.0 - 3.0)
            beta -- brightness increment (0 - 100)
        Returns:
            result -- image with adjustments applied
        '''
        result = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for i in range(3):
            result[:, :, i] = cv2.add(cv2.multiply(img[:, :, i], alpha), beta)
        return result

    def _loop_video(self):
        '''If self.loop_video == True - repeat video infinitely.'''
        if isinstance(self.source, str) and (self.loop_video):
            video_len = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cur_frame = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
            print('frame: {}/{}'.format(cur_frame, video_len))
            if self.loop_video and (cur_frame == video_len):
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _init_run(self):
        '''Initializes output and trackbars.'''
        self.count = 0
        def nothing(x):
            pass
        
        cv2.namedWindow(self.h, cv2.WINDOW_NORMAL)
        cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("screen", cv2.WND_PROP_FULLSCREEN, \
                              cv2.WINDOW_FULLSCREEN)
        cv2.createTrackbar('fit width', self.h, int(self.width * 2),\
                           self.width * 4, nothing)
        cv2.createTrackbar('mask centre', self.h, int(self.MASK_CENTRE * 100),\
                           100, nothing)
        cv2.createTrackbar('mask bottom', self.h, int(self.MASK_BOTTOM * 100),\
                           100, nothing)
        cv2.createTrackbar('mask side', self.h, 150, 200, nothing)
        cv2.createTrackbar('mask blend', self.h, int(self.MASK_BLEND * 1000),\
                           1000, nothing)
        cv2.createTrackbar('image x', self.h, int(self.width / 2), self.width,\
                             nothing)
        cv2.createTrackbar('image y', self.h, int(self.height / 2), \
                           self.height, nothing)
        cv2.createTrackbar('projections', self.h, 4, 4, nothing)
        cv2.createTrackbar('loop video', self.h, self.loop_video, 1, nothing)
        cv2.createTrackbar('Track Faces', self.h, 0, 1, nothing)
        cv2.createTrackbar('  GrabCut iters', self.h, 0, 5, nothing)
        cv2.createTrackbar('contrast', self.h, 10, 30, nothing)
        cv2.createTrackbar('brightness', self.h, 0, 100, nothing)

    def _get_trackbar_values(self):
        '''Refreshes variables with Trackbars positions.
        Returns: 
            self.pos -- positions, states and coeficients dict
        '''
        self.pos['scr_width'] = max(cv2.getTrackbarPos('fit width', self.h), \
                                    self.pyramid_size)
        self.scr_centre_x = self.pos['scr_width'] / 2
        self.pos['m_cntr'] = cv2.getTrackbarPos('mask centre', self.h) / 100
        self.pos['m_btm'] = cv2.getTrackbarPos('mask bottom', self.h)  / 100
        self.pos['m_side'] = cv2.getTrackbarPos('mask side', self.h) / 100
        self.pos['m_blend'] = cv2.getTrackbarPos('mask blend', self.h) / 1000
        self.pos['i_x'] = cv2.getTrackbarPos('image x', self.h) - \
                                                self.width / 2
        self.pos['i_y'] = cv2.getTrackbarPos('image y', self.h) - \
                                                self.height / 2
        self.pos['projections'] = cv2.getTrackbarPos('projections', self.h)
        self.loop_video = cv2.getTrackbarPos('loop video', self.h)
        self.pos['tracking_on'] = cv2.getTrackbarPos('Track Faces', self.h)
        self.pos['gc_iters'] = cv2.getTrackbarPos('  GrabCut iters', self.h)
        self.pos['contrast'] = cv2.getTrackbarPos('contrast', self.h) / 10
        self.pos['brightness'] = cv2.getTrackbarPos('brightness', self.h)

    def _apply_settings(self, frame):
        '''Apply custom settings received from Trackbars (in self.pos).
        Args:
            frame -- original frame
        Returns:
            result -- modified frame according to users adjustments
        '''
        result = frame.copy()
        self._get_trackbar_values()

        result = self._brightness_contrast(result, self.pos['contrast'], \
                                   self.pos['brightness'])

        # Translate image to (i_x, i_y)
        if (self.pos['i_x'] != 0)or(self.pos['i_y'] != 0):
            result = self._translate(result, int(self.pos['i_x']), \
                                    int(self.pos['i_y']))
        # Apply face detection mask (if ON)
        if self.pos['tracking_on']:
            tr_mask = self._track_faces(result)
            # GrabCut face(fg) extraction
            if self.pos['gc_iters'] > 0:
                gc_mask = self._grab_cut(img=result, \
                                         rect=self.gc_rect, \
                                         iters=self.pos['gc_iters'])
                result = self._apply_mask(result, gc_mask)
            else:
                result = self._apply_mask(result, tr_mask)

        # Create/Apply triangle mask
        self._create_triangle_mask(side=self.pos['m_side'], \
                                    centre=self.pos['m_cntr'], \
                                    bottom=self.pos['m_btm'])
        result = self._apply_mask(result, self.mask)
        return result

    def _rotate_projection(self, projection):
        '''Create 4(by default) projections rotated by 90 deg. CCW
        Args:
            projection -- processed frame with triangle mask applied
        Returns resulting proj. into self.screen var.
        '''
        self.screen = np.zeros((self.pyramid_size, self.pos['scr_width'], 3),\
                                np.uint8)
        for i in range(self.pos['projections']):
            self._add_projection(projection, blend=self.pos['m_blend'])
            self.screen = self._rotate(self.screen, -90, self.scr_centre_x, \
                                       self.scr_centre_y)

    def _substract_bg(self, frame):
        '''Apply Background Substraction from frame.
        Args:
            frame -- current frame
        Returns: 
            fgmask -- foreground mask
        '''
        if not self._bs_mog2_flag:
            # Create Background Substractor instance
            self._bs_mog2 = cv2.createBackgroundSubtractorMOG2(history=1000,\
                                          varThreshold=25,\
                                          detectShadows=False)
            self._bs_mog2_flag = True
        # Get FGMASK with MOG2
        gray = self._img_to_gray(frame)
        fgmask = self._bs_mog2.apply(gray, learningRate=self.pos['BS_rate'])
        # Elliptical Kernel for morphology func
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,( \
                                self.pos['k_size'], self.pos['k_size']))
        # Open (remove white points from the background)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        # Close (remove black points from the object)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        # Dilation alg (increases white regions size)
        fgmask = cv2.dilate(fgmask, kernel, iterations=self.pos['iters'])        
        return fgmask

    @staticmethod
    def _grab_cut(img, rect=None, iters=2):
        '''GrabCut image segmentation. Background identification
        Args:
            img -- image (frame) to processing
            rect -- rectangular area to be segmented. Tuple (x, y, w, h)
            iters -- algorithm iterations
        Returns:
            gc_mask -- mask of foreground
        '''
        # Create additional args required for GrabCut
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        height, width = img.shape[:2]
        mask = np.zeros((height // 2, width // 2), np.uint8)
        cv2.grabCut(cv2.pyrDown(img), mask, tuple(x // 2 for x in rect), \
                                bgdModel, fgdModel, iters, \
                                cv2.GC_INIT_WITH_RECT)
        # Substitutes all bg pixels(0,2) with sure background (0)
        gc_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') 
        return cv2.pyrUp(gc_mask)

    @staticmethod
    def _faces_to_gc_rect(faces, default_val=0):
        '''Scale up first found face coords by 2 times.
        Args:
            faces -- list of np.array([x,y,w,h]) coords of faces detected
            default_val -- Flag: process faces list or used def value instead
        Returns:
            (x, y, w, h) -- scaled up coords in tuple format
        '''
        if not default_val:
            M = np.array([[1, 0, -0.5, 0],              # Scale matrix
                          [0, 1, 0 , -0.5],
                          [0, 0, 2, 0],
                          [0, 0, 0, 2]])
            v = faces[0]                                # Face coords
            scaled_rect = np.inner(M, v).astype(int)    # Inner product M * v    
            return tuple(x for x in scaled_rect)        # Convert to tuple
        return tuple(x for x in faces[0])

    def _track_faces(self, img):
        '''Apply oval mask over faces detected in the image.
        Args:
            img -- image
        Returns:
            self.faces -- list of faces detected
            self.gc_rect -- coords. tuple for GrabCut algorithm (x, y, w, h)
            fgmask -- binary mask with faces highlighted with oval
        '''
        faces = self._detect_faces(img)
        if faces != []:                              # Face coords detected
            self.faces = faces
            self.gc_rect = self._faces_to_gc_rect(faces)
        else:                                        # Default values
            self.faces = self.def_face
            self.gc_rect = self._faces_to_gc_rect(self.faces, 1)
        fgmask = np.zeros((self.height, self.width, 3), np.uint8)
        fgmask = self._draw_ellipse(fgmask, self.faces)
        fgmask = self._img_to_gray(fgmask)
        return fgmask

    def _detect_faces(self, img):
        '''Detects faces on the image.
        Args:
            img -- image
        Returns:
            faces -- list of np.array([x,y,w,h]) face coordinates
        '''
        if not self._face_cascade.empty():
            gray = self._img_to_gray(img)
            faces = self._face_cascade.detectMultiScale(gray, \
                                                scaleFactor=1.3, \
                                                minNeighbors=4, \
                                                minSize=(30, 30), \
                                                flags=cv2.CASCADE_SCALE_IMAGE)
            if len(faces) == 0:
                return []
            return faces

    def _create_triangle_mask(self, side=MASK_SIDE, centre=MASK_CENTRE, \
                              bottom=MASK_BOTTOM):
        '''Creates triangle mask. Assigns to self.mask and its inverse to
            self.mask_inv.
        Args:
            side -- scale factor of hypotenuse of triangle mask
            centre -- centre of the mask in SCREEN
            bottom -- scale factor of legs of triangle mask
        Returns:
            self.masks -- original mask
            self.mask_inv -- mask inverse
        '''
        height = self.height
        width = self.width
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
        result = np.zeros((self.height, self.width, 3), np.uint8)
        # Create traiangle
        result = cv2.fillConvexPoly(result, pts, (255, 255, 255), 1)
        # Convert to GRAY
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        self.mask = result
        self.mask_inv = cv2.bitwise_not(self.mask)

    @staticmethod
    def _apply_mask(img, mask):
        '''Apply mask to image'''
        return cv2.bitwise_and(img, img, mask=mask)

    def _add_projection(self, projection, blend=MASK_BLEND):
        '''Adds PROJECTION to bottom centre of SCREEN.
        Args:
            projection -- processed frame
            blend -- position of the triang. mask on the frame
        Returns:
            self.screen -- image ready to project on Pyramide
            '''
        p_height, p_width = projection.shape[:2]
        #Extract ROI
        x = self.scr_centre_x - p_width / 2    
        y = self.scr_centre_y - p_height * blend
        roi = self.screen[y : y + p_height, x : x + p_width]
        # Black-out the area of projection in ROI
        roi_bg = self._apply_mask(roi, self.mask_inv)
        # Add projection to ROI
        dst = cv2.add(roi_bg, projection)
        # Apply ROI to SCREEN
        self.screen[y : y + p_height, x : x + p_width] = dst

    def _debugger_mode(self, frame, projection):
        '''Adds two additional windows for debugging and adjustments.
        Args:
            frame -- original frame from video input
            projection -- processed frame
        '''
        if self.DEBUGGER_MODE:
            frame = self._draw_rect(frame, [self.gc_rect])
            cv2.imshow('original', frame)
            cv2.moveWindow('original', 0, 0)
            cv2.imshow('result', projection)
            cv2.moveWindow('result', 0, self.height)
            self._debugger_off = False
        else:
            if not self._debugger_off:
                cv2.destroyWindow('original')
                cv2.destroyWindow('result')
                self._debugger_off = True

    @staticmethod
    def _draw_rect(img, faces):
        '''Draw BLUE rectangle on img.'''
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return img

    @staticmethod
    def _draw_ellipse(img, faces):
        '''Draws solid filled white ellipse fitted into rectangle area.
        Args:
            img -- image (frame)
            faces -- ist of np.array([x,y,w,h]) face coordinates
        Returns:
            result -- img with white elipses
        '''
        for (x, y, w, h) in faces:
            result = cv2.ellipse(img, (x + w // 2, y + h // 2), \
                                    (w // 2, int(h / 1.5)), \
                                    0, 0, 360, (255, 255, 255), -1)
        return result

    @staticmethod
    def _img_to_gray(img):
        '''Converts img (frame) to Grayscale, adjust contrast.'''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(img) # Adj. contrast

    @staticmethod
    def _translate(img, x_dist, y_dist):
        rows, cols = img.shape[:2]
        M = np.float32([[1, 0, x_dist], [0, 1, y_dist]])
        result = cv2.warpAffine(img, M, (cols, rows))
        return result

    @staticmethod
    def _rotate(img, angle, anchor_x, anchor_y):
        '''Rotates image about anchor point
        Args:
            img -- image
            angel -- angle to rotate
            anchor_x, anchor_y -- anchor coordinates to rotate about.
        Returns:
            result -- rotated image (no scaling)
        '''
        height, width = img.shape[:2]
        M = cv2.getRotationMatrix2D((anchor_x, anchor_y), angle, 1)
        result = cv2.warpAffine(img, M, (width, height))
        return result    

    @staticmethod
    def _scale(img, ratio):
        '''Uniform scale'''
        result = cv2.resize(img, None, fx=ratio, fy=ratio)
        return result

    def _show(self, img):
        '''Shows img in the 'show' window'''
        cv2.imshow('show', img)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:     # ESC key
            cv2.destroyWindow('show')

if __name__ == '__main__':
    #ghost = Ghost('/home/chip/pythoncourse/hologram2/test3.mp4')
    ghost = Ghost()

    #path = getcwd()
    #ghost.set_output(path+'/out.avi')

    ghost.run()
