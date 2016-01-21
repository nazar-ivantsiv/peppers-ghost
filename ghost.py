#!bin/python
"""
==============================================================================
Pepper's Ghost video processor.

Author:
    Nazar Ivantsiv
Mentors:
    Nazar Grabovskyy
    Igor Lushchyk

Creates pyramid like projection of input video (camera or file), ready to
use with pseudo holographic pyramid.
Source -- video camera or video file
Output -- monitor, VLC stream or video file

USAGE:
    'SETTINGS' window gives you ability to change different Ghost instance
attributes. The final result is viewed on output_img window. You can switch it
into Fullscreen mode by dragging it into desired monitor and pressing 'f' key.


Key 'q' - To quit
Key 'd' - Debugger windows (on/off)
Key 'f' - Fullscreen mode (on/off)
Key 'o' - Open video output (VLC stream or video file file)
Key 'r' - Release video output
==============================================================================
"""

from __future__ import division
from os import getcwd
from time import time, sleep
from datetime import datetime
import pdb

import numpy as np
import cv2

from modules.capture_thread import Capture
from modules.gui import Gui
from modules.output_vlc import Output
from modules.segmentation import FaceExtraction
from modules.im_trans import apply_mask
from modules.im_trans import brightness_contrast
from modules.im_trans import create_triangle_mask
from modules.im_trans import draw_rect
from modules.im_trans import draw_ellipse
from modules.im_trans import fit_into
from modules.im_trans import img_to_gray
from modules.im_trans import rotate
from modules.im_trans import scale
from modules.im_trans import translate


class Ghost(object):
    """Pepper's Ghost video processor."""

    def __init__(self, source=0):
        self.cap = Capture(source)              # Input instance:
                                                # def. source == 0 (webcamera)                                                
        self.out = Output()
        self.height = self.cap.height           # Frame height
        self.width = self.cap.width             # Frame width
        self.scr_height = self.height * 2       # Scr height
        self.scr_width = self.scr_height#self.width * 2         # Scr width
        self.scr_centre_x = self.height#self.width          # Scr centre y
        self.scr_centre_y = self.height         # Scr centre x
        self.fps = self.cap.fps
        self.ORD_DICT = {'q':ord('q'), \
                         'd':ord('d'), \
                         'f':ord('f'), \
                         'o':ord('o'), \
                         'r':ord('r'), \
                         'p':ord('p')}
        self.face_ex = FaceExtraction(self.height, self.width)
        self.pos = {}                           # Dict with trackbars values
#        self.pos['scr_width'] = self.width * 2  # A crutch to expand black area 
                                                # around the output_img.
                                                # (Not needed if we use VLC client.)
        self.pos['m_cntr'] = 0                  # Centre of the mask in output_img
        self.pos['m_btm'] = 1                   # Mask bottom corner position
        self.pos['m_side'] = 1.5                # Mask side corners pos
        self.pos['m_y_offset'] = 1              # Offset ratio of y mask coord.
        self.pos['i_x'] = int(self.width / 2)   # frame x pos
        self.pos['i_y'] = int(self.height / 2)  # frame y pos
        self.pos['scale'] = 1                   # frame scale factor
        self.pos['angle'] = 90                  # angle relation between projections
        self.pos['proj_num'] = 1                # projections qty (def. 4)
        self.pos['loop_video'] = 1              # on/off
        self.pos['tracking_on'] = 0             # FaceTracking on/off
        self.pos['gc_iters'] = 0                # GrabCut iterations
        self.pos['contrast'] = 1                # contrast adj.
        self.pos['brightness'] = 0              # brightness adj.
        self.gui = Gui(self.height, self.width, self.pos) # GUI instance
        self.counter = 0                        # Frame counter

    def run(self):
        """Video processing."""
        self.start = time()
        while self.cap.is_opened():
            frame = self.cap.next_frame()
            # Create output_img with all settings/segmentations applied to frame
            frame_mod = self._apply_settings(frame)
            output_img = self._create_output_img(frame=frame_mod, \
                                         proj_num=self.pos['proj_num'], \
                                         y_offset_ratio=self.pos['m_y_offset'], \
                                         angle=self.pos['angle'])
            if self.out.is_opened:               # Send output_img to output
                self.out.write(output_img)

            # Operation routines
            self.print_fps(self.gui.C_HDR)
            self.gui.preview(output_img)       # Preview into Output window
            key_pressed = cv2.waitKey(1) & 0xFF
            if not self._process_key(key_pressed):
                break
            self.counter += 1
            self.end = time()
            self.fps = self.calc_fps()
        self.stop()

    def calc_fps(self):
        fps = round(self.counter / (self.end - self.start), 2)
        if self.counter == 100:
            self.counter = 0
            self.start = self.end
        return fps

    def print_fps(self, window_hrd='FPS'):
        canvas = np.zeros((70, 500, 3), np.uint8)
        cv2.putText(canvas, 'FPS: {0:.1f}'.format(self.fps), \
                    (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, \
                    (255,255,255), 3, cv2.LINE_AA)
        cv2.imshow(window_hrd, canvas)

    def stop(self):
        """Release input/output if job is finished. And exit GUI."""
        self.cap.release()
        self.out.release()
        self.gui.exit()

    def _apply_settings(self, frame):
        """Apply custom settings received from GUI (in self.pos).
        Args:
            frame -- original frame
        Returns:
            result -- modified frame according to users adjustments
        """
        # Translate frame to (i_x, i_y)
        if (self.pos['i_x'] != 0)or(self.pos['i_y'] != 0):
            frame = translate(frame, int(self.pos['i_x']), \
                                    int(self.pos['i_y']))
        # Scale frame
        if self.pos['scale'] != 1:
            frame_scaled = scale(frame, self.pos['scale'])
            frame = fit_into(frame_scaled, self.height, self.width)
        # Adjust brightness/contrast
        result = frame.copy()
        result = brightness_contrast(result, self.pos['contrast'], \
                                   self.pos['brightness'])
        # Apply face detection mask (if ON)
        if self.pos['tracking_on']:
            tr_mask = self.face_ex.track_faces(frame)
            # GrabCut face(fg) extraction
            if self.pos['gc_iters'] > 0:
                gc_mask = self.face_ex.gc_mask( img=frame, \
                                                iters=self.pos['gc_iters'])
                result = apply_mask(result, gc_mask)
            else:
                result = apply_mask(result, tr_mask)
        # Create triangle mask
        # Add decorator here
        self.mask = create_triangle_mask(height=self.height, \
                                        width=self.width, \
                                        side=self.pos['m_side'], \
                                        centre=self.pos['m_cntr'], \
                                        bottom=self.pos['m_btm'])
#        self.scr_width = self.pos['scr_width']
#        self.scr_centre_x = self.scr_width // 2
        self.cap.loop_video = self.pos['loop_video']
        return result

    def _create_output_img(self, frame, proj_num=4, y_offset_ratio=1, angle=90):
        """Create projections rotated by 'angle' deg. CCW
        Args:
            frame -- processed frame
            proj_num -- projections qty
            y_offset_ratio -- mask y coord offset ratio
        Returns:
            output_img -- resulting output_img.
        """
        # Create blank output_img
        output_img = np.zeros((self.scr_height, self.scr_width, 3), np.uint8)
        # Calculate frame position in the output_img
        frame_x = self.scr_centre_x - self.width // 2
        frame_y = self.scr_centre_y - self.height * y_offset_ratio
        # Apply triangle mask on projection
        projection = apply_mask(frame, self.mask)
        # Apply projection to Bottom Centre of output_img
        output_img[frame_y : frame_y + self.height, \
                    frame_x : frame_x + self.width] = projection
        # Add Top projection
        output_img = cv2.add(output_img, cv2.flip(output_img, -1))
        # Add Left and Right projections
        output_img = cv2.add(output_img, cv2.flip(cv2.transpose(output_img), 1))
        return output_img

    def _process_key(self, key_pressed):
        """Method called from while loop in run(). Porcesses the key pressed.
        returns 0 if EXIT button (def. 'q') is pressed. Else returns 1.
        Args:
            key_pressed -- ord value of key pressed
        Reurns:
            1 -- take an action and go ahead i the loop
            0 -- take an action and BREAK the loop
        """
        if key_pressed == self.ORD_DICT['q']:         # Wait for 'q' to exit
            return 0
        elif key_pressed == self.ORD_DICT['d']:       # Debugger windows(on/off)
            self.gui.toggle_debugger()
        elif key_pressed == self.ORD_DICT['f']:       # Fullscreen on/off
            self.gui.toggle_fullscreen()
        elif key_pressed == self.ORD_DICT['o']:       # Set output
            self.out.set_output(self.scr_height, self.scr_width)
        elif key_pressed == self.ORD_DICT['r']:       # Release output
            self.out.release()
        elif key_pressed == self.ORD_DICT['p']:       # Preview on/off
            self.gui.toggle_preview()
        return 1


if __name__ == '__main__':

    import cProfile
    import pstats

    #ghost = Ghost('/home/chip/pythoncourse/hologram2/test2.mp4')
    ghost = Ghost(0)

    #path = getcwd()
    #ghost.set_output(path+'/out.avi')

    cProfile.run('ghost.run()')
    #ghost.run()
