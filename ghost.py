#!bin/python

# -*- coding: utf-8 -*-
from __future__ import division
from os import getcwd

import numpy as np
import cv2

from matplotlib import pyplot as plt


HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

class Ghost(object):
    '''Pepper's Ghost video processor.
    Creates pyramid like projection of input video (camera or file), ready to
    use with pseudo holographic pyramid.
    Args:
        source: video camera or file
    '''
    DEBUGGER_MODE = 1 #0
    MASK_CENTRE = 0 #0.5
    MASK_BOTTOM = 1 #0.33
    MASK_BLEND = 1 #0.915

    def __init__(self, source=0):
        '''Args:
            source: camera (0, 1) or vieofile
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
        self._fgbg_flag = False                 # Flag: BS instance created
        self._face_cascade_flag = False         # Flag: Cascade clsfr for GC
        self._debugger_off = not self.DEBUGGER_MODE # Flag: debugger status

    def run(self):
        '''Starts video processor.'''
        self._init_run()
        cap = self._cap
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                # Create projection with all the masks applied to original frame
                projection = self._apply_settings(frame)                   
                self._rotate_projection(projection)
                if self._out != None:               # Save video to file
                    self._out.write(self.screen)
                cv2.imshow('screen', self.screen)   # Output SCREEN to window
                self._loop_video()                  # Loop the video
                if cv2.waitKey(1) & 0xFF == ord('q'): # Wait for 'q' to exit
                    break
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
                            (self.pyramid_size, self.pyramid_size))

    def stop(self):
        '''Release everything if job is finished. And close the window.'''
        self._cap.release()
        if self._out != None:
            self._out.release()
        cv2.destroyAllWindows()

    def _loop_video(self):
        # self.source == video path
        if isinstance(self.source, str) and (self.loop_video):
            video_len = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cur_frame = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
            print('frame: {}/{}'.format(cur_frame, video_len))
            if self.loop_video and (cur_frame == video_len):
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _init_run(self):
        '''Initializes output and adjustments'''
        self.count = 0
        def nothing(x):
            pass
        cv2.namedWindow(self.h, cv2.WINDOW_NORMAL)
        cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("screen", cv2.WND_PROP_FULLSCREEN, \
                              cv2.WINDOW_FULLSCREEN)
        # Create trackbars
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
        #cv2.createTrackbar('image scale', self.h, 10, 19, nothing)
        cv2.createTrackbar('loop video', self.h, self.loop_video, 1, nothing)
        cv2.createTrackbar('Track Faces', self.h, 0, 1, nothing)
        cv2.createTrackbar('  GrabCut iters', self.h, 0, 5, nothing)
        cv2.createTrackbar('BS:', self.h, 0, 1, nothing)
        cv2.createTrackbar('  BS learn.rate', self.h, 20, 50, nothing)
        cv2.createTrackbar('  dilation kernel size', self.h, 5, 20, nothing)
        cv2.createTrackbar('  dilation iters', self.h, 3, 10, nothing)
        cv2.createTrackbar('debugger', self.h, self.DEBUGGER_MODE, 1, nothing)

    def _get_trackbar_values(self):
        '''Refreshes variables with Trackbars positions (self.pos - var dict)'''
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
        #i_ratio = cv2.getTrackbarPos('image scale', self.h) / 10 or 1
        self.loop_video = cv2.getTrackbarPos('loop video', self.h)
        self.pos['tracking_on'] = cv2.getTrackbarPos('Track Faces', self.h)
        self.pos['gc_iters'] = cv2.getTrackbarPos('  GrabCut iters', self.h)
        self.pos['BS_on'] = cv2.getTrackbarPos('BS:', self.h)
        self.pos['BS_rate'] = cv2.getTrackbarPos('  BS learn.rate', self.h) / 10000
        self.pos['k_size'] = cv2.getTrackbarPos('  dilation kernel size', \
                                                self.h) or 1
        self.pos['iters'] = cv2.getTrackbarPos('  dilation iters', self.h)
        self.pos['debugger'] = cv2.getTrackbarPos('debugger', self.h)

    def _apply_settings(self, frame):
        '''Apply custom settings from Trackbars.
        Args:
            frame: original frame
        Returns:
            result: modified frame according to user settings 
        '''
        result = frame.copy()
        self._get_trackbar_values()
        # Translate image to (i_x, i_y)
        if (self.pos['i_x'] != 0)or(self.pos['i_y'] != 0):
            result = self._translate(result, int(self.pos['i_x']), \
                                    int(self.pos['i_y']))
        # Background Substraction (if ON)
        if self.pos['BS_on']:
            bs_mask = self._substract_bg(result)
            result = self._apply_mask(result, bs_mask)
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
        '''Create FOUR projections rotated by -90 deg'''
        self.screen = np.zeros((self.pyramid_size, self.pos['scr_width'], 3),\
                                np.uint8)
        for i in range(self.pos['projections']):
            self._add(projection, blend=self.pos['m_blend'])
            self.screen = self._rotate(self.screen, -90, self.scr_centre_x, \
                                       self.scr_centre_y)

    def _substract_bg(self, frame):
        '''Apply Background Substraction on frame.
        Args:
            frame: current frame
        Returns: 
            fgmask: foreground mask
        '''
        if not self._fgbg_flag:
            # Create Background Substractor instance
            self._fgbg = cv2.createBackgroundSubtractorMOG2(history=1000,\
                                          varThreshold=25,\
                                          detectShadows=False)
            self._fgbg_flag = True
        # Get FGMASK with MOG2
        gray = self._img_to_gray(frame)
        fgmask = self._fgbg.apply(gray, learningRate=self.pos['BS_rate'])
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

    def _grab_cut(self, img, rect, iters=5):
        ''' '''
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        height, width = img.shape[:2]
        mask = np.zeros((height // 2, width // 2), np.uint8)
        cv2.grabCut(cv2.pyrDown(img), mask, tuple(x // 2 for x in rect), \
                                bgdModel, fgdModel, iters, \
                                cv2.GC_INIT_WITH_RECT)
        # Substitutes all bg pixels(0,2) with sure background (0)
        gc_mask = np.where((mask==2) | (mask==0), 0, 1).astype('uint8') 
        return cv2.pyrUp(gc_mask)

    def _faces_to_gc_rect(self, faces, default_val=0):
        '''Scale up face rect 2 times. Convert to tuple'''
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
            img: image
        Returns:
            self.face_x: first detected face X coord.
            self.face_y: first detected face Y coord.
            img: image with the oval mask around the face
        '''
        if not self._face_cascade_flag:
            # Create classifier instance
            self._face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
            self._face_cascade_flag = True
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
            img: image
        Returns:
            faces - list of (x,y,w,h) face coordinates
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

    def _create_triangle_mask(self, side=1.5, centre=MASK_CENTRE, bottom=MASK_BOTTOM):
        '''Creates triangle mask. And saves to self.mask and self.mask_inv.
        Args:
            height: original image HEIGHT
            width: original image WIDTH
            scale_x: scale mask height
            scale_y: scale mask width
            centre: move mask centre by Y axis
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

    def _add(self, projection, blend=MASK_BLEND):
        '''Adds PROJECTION to bottom centre of SCREEN'''
        p_height, p_width = projection.shape[:2]
        #Extract ROI
        x = self.scr_centre_x - p_width / 2    
        y = self.scr_centre_y - p_height * blend
        roi = self.screen[y : y + p_height, x : x + p_width]
        # Black-out the area of projection in ROI
        roi_bg = self._apply_mask(roi, self.mask_inv)
        # Add WINDOW to ROI
        dst = cv2.add(roi_bg, projection)
        # Apply ROI to SCREEN
        self.screen[y : y + p_height, x : x + p_width] = dst

    def _debugger_mode(self, frame, projection):
        if self.pos['debugger']:
            frame = self._draw_rect(frame, [self.gc_rect])
            cv2.imshow('original', frame)
            cv2.imshow('result', projection)
            self._debugger_off = False
        else:
            if not self._debugger_off:
                cv2.destroyWindow('original')
                cv2.destroyWindow('result')
                self._debugger_off = True


    @staticmethod
    def _draw_rect(img, faces):
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return img

    @staticmethod
    def _draw_ellipse(img, faces):
        '''Draws solid filled white ellipse fitted into rectangle area.
        Args:
            img: image
            faces: list of (x,y,w,h) face coordinates (rectangles)
        Returns:
            result: img with white elipses
        '''
        for (x, y, w, h) in faces:
            img = cv2.ellipse(img, (x + w // 2, y + h // 2), \
                                    (w // 2, int(h / 1.5)), \
                                    0, 0, 360, (255, 255, 255), -1)
        return img

    @staticmethod
    def _img_to_gray(img):
        '''Converts frame to Grayscale, adjust contrast.'''
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
    def _scale(img, ratio):
        '''Uniform scale'''
        result = cv2.resize(img, None, fx=ratio, fy=ratio)
        return result

    def _show(self, img):
        cv2.imshow('show', img)

        k = cv2.waitKey(0) & 0xFF

        if k == 27:     # ESC key
            cv2.destroyWindow('show')

    @staticmethod
    def _show_plt(img):
        plt.imshow(img, cmap='gray', interpolation='bicubic')
        plt.show()


if __name__ == '__main__':
    #ghost = Ghost('/home/chip/pythoncourse/hologram2/test.mp4')
    ghost = Ghost()

    #path = getcwd() + '/out.avi'
    #ghost.set_output(path)

    ghost.run()
