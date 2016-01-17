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

import numpy as np
import cv2

from modules.capture import Capture
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
        self.ORD_DICT = {'q':ord('q'), \
                         'd':ord('d'), \
                         'f':ord('f'), \
                         'o':ord('o'), \
                         'r':ord('r')}
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
        self.pos['proj_num'] = 4                # projections qty (def. 4)
        self.pos['loop_video'] = 1              # on/off
        self.pos['tracking_on'] = 0             # FaceTracking on/off
        self.pos['gc_iters'] = 0                # GrabCut iterations
        self.pos['contrast'] = 1                # contrast adj.
        self.pos['brightness'] = 0              # brightness adj.
        self.gui = Gui(self.height, self.width, self.pos) # GUI instance

    def run(self):
        """Video processing."""
        while self.cap.is_opened():
            start = time()
            frame = self.cap.next_frame()
            # Create output_img with all settings/segmentations applied to frame
            frame_mod = self._apply_settings(frame)
            output_img = self._create_output_img(frame=frame_mod, \
                                         proj_num=self.pos['proj_num'], \
                                         y_offset_ratio=self.pos['m_y_offset'], \
                                         angle=self.pos['angle'])
            if self.out.is_opened:               # Save video to output
                self.out.write(output_img)#self.out.write(scale(output_img, 0.7))

            # Operation routines
            self.gui.preview(output_img)       # Preview into output_img window
            key_pressed = 0xFF & cv2.waitKey(1)
            if not self._process_key(key_pressed, start):
                break
            if self.gui.DEBUGGER_MODE:          # Shows debugger win if ON
                frame = self._add_fps( frame, start )
                frame = draw_rect( frame, self.face_ex.faces )
                self.gui.debugger_show(frame, frame_mod)
        self.stop()

    def stop(self):
        """Release input/output if job is finished. And exit GUI."""
        self.cap.release()
        self.out.release()
        self.gui.exit()

    def _add_fps(self, frame, start):
        fps = 1 / (time() - start)
        cv2.putText(frame, 'fps: {0:.1f}'.format(fps), \
                    (10, self.cap.height - 50), cv2.FONT_HERSHEY_PLAIN, 3, \
                    (255,255,255), 2, cv2.LINE_AA)
        return frame

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

    def _create_output_img(self, frame, proj_num=4, y_offset_ratio=1, angle=-90):
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
        frame_height, frame_width = frame.shape[:2]             # Proj. dimentions
        frame_x = self.scr_centre_x - frame_width // 2            # 
        frame_y = self.scr_centre_y - frame_height * y_offset_ratio        # 
        # Create projection with triangle mask
        projection = apply_mask(frame, self.mask)
        mask_inv = cv2.bitwise_not(self.mask)
        for i in range(proj_num):
            #Extract ROI
            roi = output_img[frame_y : frame_y + frame_height, frame_x : \
                                frame_x + frame_width]
            # Black-out foreground in ROI
            roi_bg = apply_mask(roi, mask_inv)
            # Add projection to ROI background
            dst = cv2.add(roi_bg, projection)
            # Put ROI on output_img, in place
            output_img[frame_y : frame_y + frame_height, \
                        frame_x : frame_x + frame_width] = dst
            output_img = rotate(output_img, angle, self.scr_centre_x, self.scr_centre_y)
        return output_img

    def _process_key(self, key_pressed, start):
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
        elif key_pressed == self.ORD_DICT['o']:       # Set output file
            if self.cap.source != -1:       # Is NOT a video file
                fps = int(round(1 / (time() - start)))
            else:
                fps = self.cap.fps          # Use video file fps
            self.out.set_output(fps, self.scr_height, \
                                self.scr_width)
        elif key_pressed == self.ORD_DICT['r']:       # Release output
            self.out.release()
        return 1

if __name__ == '__main__':

    import cProfile
    #ghost = Ghost('/home/chip/pythoncourse/hologram2/test2.mp4')
    ghost = Ghost(0)

    #path = getcwd()
    #ghost.set_output(path+'/out.avi')

    cProfile.run('ghost.run()')
    #ghost.run()
