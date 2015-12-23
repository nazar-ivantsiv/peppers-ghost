#!bin/python
"""
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
Key 'f' - Fullscreen mode (on/off)

==============================================================================
"""

from __future__ import division
from os import getcwd

import numpy as np
import cv2

# GUI for Peppers Ghost
from modules import gui

# Features for CascadeClassifier (frontal face)
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

class Ghost(object):
    """Pepper's Ghost video processor."""

    def __init__(self, source=0):
        """Args:
            source -- camera (0, 1) or vieofile
        """
        self.source = source                    # Input source num/path
        self.h = 'SETTINGS'
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            self._cap.open()                    # Input source instance
        self._out = None                        # Output source instance(file)
        self.height = int(self._cap.get(4))     # Frame height
        self.width = int(self._cap.get(3))      # Frame width
        self.scr_height = self.height * 2       # Scr height
        self.scr_width = self.width * 2         # Scr width
        self.scr_centre_x = self.width          # Scr centre y
        self.scr_centre_y = self.height         # Scr centre x
            # Faces detected (by default - rect in the centre)
        x = self.width // 2                     # Frame centre x
        y = self.height // 2                    # Frame centre y
        a = x // 2
        self.faces = [np.array([x - a, y - a, x, y])]
        self.gc_rect = (x - a, y - a, x, y)     # GrabCut default rect
        self.def_face = self.faces              # Default value (rect in centre)
        self._face_cascade_flag = False         # Flag: Cascade clsfr for GC

        self.gui = gui.Gui(self.height, self.width)

    def run(self):
        """Video processing."""
        cap = self._cap
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                # Create projection with all available masks applied to frame
                projection = self._apply_settings(frame)
                self._rotate_projection(projection)

                if self._out != None:               # Save video to file
                    self._out.write(self.screen)
                cv2.imshow(self.gui.S_HDR, self.screen)   # Output SCREEN to window
# MOVE THIS TO OUTPUT cls
                self._loop_video()                  # Loop the video 

                key_pressed = 0xFF & cv2.waitKey(1)
                if key_pressed == ord('q'):         # Wait for 'q' to exit
                    break
                elif key_pressed == ord('d'):       # Debugger windows(on/off)
                    self.gui.toggle_debugger()
                elif key_pressed == ord('f'):       # Fullscreen on/off
                    self.gui.toggle_fullscreen()

                if self.gui.DEBUGGER_MODE:          # Shows debugger win if ON
                    frame = self._draw_rect(frame, [self.gc_rect])
                    self.gui.debugger_show(frame, projection)
            else:
                break
        self.stop()

    def set_output(self, file_path):
        """Define the codec and create VideoWriter object to output video.
        Args:
            file_path: filename or path to output file
        """
        print('Writing video to: {}'.format(file_path))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._out = cv2.VideoWriter(file_path, fourcc, 20.0, \
                            (self.width * 2, self.height * 2))
        print(self.width * 2, self.height * 2)

    def stop(self):
        """Release everything if job is finished. And close the windows."""
        self._cap.release()
        if self._out != None:
            self._out.release()
        cv2.destroyAllWindows()

    @staticmethod
    def _brightness_contrast(img, alpha, beta):
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

    def _loop_video(self):
        """If self.loop_video == True - repeat video infinitely."""
        if isinstance(self.source, str) and (self.loop_video):
            video_len = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cur_frame = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
            print('frame: {}/{}'.format(cur_frame, video_len))
            if self.loop_video and (cur_frame == video_len):
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)



    def _apply_settings(self, frame):
        """Apply custom settings received from GUI (in self.gui.pos).
        Args:
            frame -- original frame
        Returns:
            result -- modified frame according to users adjustments
        """
        self.gui.get_trackbar_values()
        self.scr_width = self.gui.pos['scr_width']
        self.scr_centre_x = self.scr_width / 2
        self.loop_video = self.gui.pos['loop_video']

        result = frame.copy()
        result = self._brightness_contrast(result, self.gui.pos['contrast'], \
                                   self.gui.pos['brightness'])
        # Translate image to (i_x, i_y)
        if (self.gui.pos['i_x'] != 0)or(self.gui.pos['i_y'] != 0):
            result = self._translate(result, int(self.gui.pos['i_x']), \
                                    int(self.gui.pos['i_y']))
        # Apply face detection mask (if ON)
        if self.gui.pos['tracking_on']:
            tr_mask = self._track_faces(frame)
            # GrabCut face(fg) extraction
            if self.gui.pos['gc_iters'] > 0:
                gc_mask = self._grab_cut(img=frame, \
                                         rect=self.gc_rect, \
                                         iters=self.gui.pos['gc_iters'])
                result = self._apply_mask(result, gc_mask)
            else:
                result = self._apply_mask(result, tr_mask)
        # Create/Apply triangle mask
        self._create_triangle_mask(side=self.gui.pos['m_side'], \
                                    centre=self.gui.pos['m_cntr'], \
                                    bottom=self.gui.pos['m_btm'])
        result = self._apply_mask(result, self.mask)
        return result

    def _rotate_projection(self, projection):
        """Create 4(by default) projections rotated by 90 deg. CCW
        Args:
            projection -- processed frame with triangle mask applied
        Returns resulting proj. into self.screen var.
        """
        self.screen = np.zeros((self.scr_height, self.scr_width,\
                                 3), np.uint8)
        for i in range(self.gui.pos['projections']):
            self._add_projection(projection, blend=self.gui.pos['m_blend'])
            self.screen = self._rotate(self.screen, -90, self.scr_centre_x, \
                                       self.scr_centre_y)

    def _substract_bg(self, frame):
        """Apply Background Substraction from frame.
        Args:
            frame -- current frame
        Returns: 
            fgmask -- foreground mask
        """
        if not self._bs_mog2_flag:
            # Create Background Substractor instance
            self._bs_mog2 = cv2.createBackgroundSubtractorMOG2(history=1000,\
                                          varThreshold=25,\
                                          detectShadows=False)
            self._bs_mog2_flag = True
        # Get FGMASK with MOG2
        gray = self._img_to_gray(frame)
        fgmask = self._bs_mog2.apply(gray, learningRate=self.gui.pos['BS_rate'])
        # Elliptical Kernel for morphology func
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,( \
                                self.gui.pos['k_size'], self.gui.pos['k_size']))
        # Open (remove white points from the background)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        # Close (remove black points from the object)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        # Dilation alg (increases white regions size)
        fgmask = cv2.dilate(fgmask, kernel, iterations=self.gui.pos['iters'])        
        return fgmask

    @staticmethod
    def _grab_cut(img, rect=None, iters=2):
        """GrabCut image segmentation. Background identification
        Args:
            img -- image (frame) to processing
            rect -- rectangular area to be segmented. Tuple (x, y, w, h)
            iters -- algorithm iterations
        Returns:
            gc_mask -- mask of foreground
        """
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
        """Scale up first found face coords by 2 times.
        Args:
            faces -- list of np.array([x,y,w,h]) coords of faces detected
            default_val -- Flag: process faces list or used def value instead
        Returns:
            (x, y, w, h) -- scaled up coords in tuple format
        """
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
        """Apply oval mask over faces detected in the image.
        Args:
            img -- image
        Returns:
            self.faces -- list of faces detected
            self.gc_rect -- coords. tuple for GrabCut algorithm (x, y, w, h)
            fgmask -- binary mask with faces highlighted with oval
        """
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
        """Detects faces on the image.
        Args:
            img -- image
        Returns:
            faces -- list of np.array([x,y,w,h]) face coordinates
        """
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

    def _create_triangle_mask(self, side=1.5, centre=0, \
                              bottom=1):
        """Creates triangle mask. Assigns to self.mask and its inverse to
            self.mask_inv.
        Args:
            side -- scale factor of hypotenuse of triangle mask
            centre -- centre of the mask in SCREEN
            bottom -- scale factor of legs of triangle mask
        Returns:
            self.masks -- original mask
            self.mask_inv -- mask inverse
        """
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
        """Apply mask to image"""
        return cv2.bitwise_and(img, img, mask=mask)

    def _add_projection(self, projection, blend=1):
        """Adds PROJECTION to bottom centre of SCREEN.
        Args:
            projection -- processed frame
            blend -- position of the triang. mask on the frame
        Returns:
            self.screen -- image ready to project on Pyramide
            """
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

    @staticmethod
    def _draw_rect(img, faces):
        """Draw BLUE rectangle on img."""
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return img

    @staticmethod
    def _draw_ellipse(img, faces):
        """Draws solid filled white ellipse fitted into rectangle area.
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

    @staticmethod
    def _img_to_gray(img):
        """Converts img (frame) to Grayscale, adjust contrast."""
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

    @staticmethod
    def _scale(img, ratio):
        """Uniform scale"""
        result = cv2.resize(img, None, fx=ratio, fy=ratio)
        return result

    def _show(self, img):
        """Shows img in the 'show' window"""
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
