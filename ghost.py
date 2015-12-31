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
    'SETTINGS' window gives you ability to change different Ghost instance
attributes. The final result is viewed on SCREEN window. You can switch it
into Fullscreen mode by dragging it into desired monitor and pressing 'f' key.


Key 'q' - To quit
Key 'd' - Debugger windows (on/off)
Key 'f' - Fullscreen mode (on/off)
Key 'o' - Open video output (write to file 'out.avi')
Key 'r' - Release video output
==============================================================================
"""

from __future__ import division
from os import getcwd

import numpy as np
import cv2

from modules.gui import Gui
from modules.capture import Capture
from modules.output import Output
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
        self.scr_width = self.width * 2         # Scr width
        self.scr_centre_x = self.width          # Scr centre y
        self.scr_centre_y = self.height         # Scr centre x

        self.face_ex = FaceExtraction(self.height, self.width)

        self.gui = Gui(self.height, self.width) # GUI instance
        self.pos = self.gui.pos                 # Dict with trackbars values

    def run(self):
        """Video processing."""
        while self.cap.is_opened():
            frame = self.cap.next_frame()
            # Create SCREEN with all settings/segmentations applied to frame
            self.refresh_values()

            frame_mod = self.apply_settings(frame)

            screen = self._create_screen(frame=frame_mod, \
                                         projections=self.pos['projections'], \
                                         blend=self.pos['m_blend'], \
                                         angle=self.pos['angle'])
            if self.out.is_opened:               # Save video to output
                self.out.write(screen)
            # Operation routines
            self.gui.preview(screen)       # Preview into SCREEN window
            key_pressed = 0xFF & cv2.waitKey(1)
            if key_pressed == ord('q'):         # Wait for 'q' to exit
                break
            elif key_pressed == ord('d'):       # Debugger windows(on/off)
                self.gui.toggle_debugger()
            elif key_pressed == ord('f'):       # Fullscreen on/off
                self.gui.toggle_fullscreen()
            elif key_pressed == ord('o'):       # Set output file
                self.out.set_output('out.avi', self.scr_height, self.scr_width)
            elif key_pressed == ord('r'):       # Release output
                self.out.release()
            if self.gui.DEBUGGER_MODE:          # Shows debugger win if ON
                frame = draw_rect(frame, self.face_ex.faces)
                self.gui.debugger_show(frame, frame_mod)
        self.stop()

    def stop(self):
        """Release input/output if job is finished. And exit GUI."""
        self.cap.release()
        self.out.release()
        self.gui.exit()

    def refresh_values(self):
        self.gui.get_trackbar_values()
        self.scr_width = self.pos['scr_width']
        self.scr_centre_x = self.scr_width // 2
        self.cap.loop_video = self.pos['loop_video']

    def apply_settings(self, frame):
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
        self.mask = create_triangle_mask(height=self.height, \
                                        width=self.width, \
                                        side=self.pos['m_side'], \
                                        centre=self.pos['m_cntr'], \
                                        bottom=self.pos['m_btm'])
        return result

    def _create_screen(self, frame, projections=4, blend=1, angle=-90):
        """Create projections rotated by 'angle' deg. CCW
        Args:
            frame -- processed frame
            projections -- projections qty
            bleng -- mask
        Returns:
            screen -- resulting screen.
        """
        # Create blank SCREEN
        screen = np.zeros((self.scr_height, self.scr_width, 3), np.uint8)
        p_height, p_width = frame.shape[:2]
        x = self.scr_centre_x - p_width / 2
        y = self.scr_centre_y - p_height * blend
        # Create projection with triangle mask
        projection = apply_mask(frame, self.mask)
        mask_inv = cv2.bitwise_not(self.mask)
        for i in range(projections):
            #Extract ROI
            roi = screen[y : y + p_height, x : x + p_width]
            # Black-out foreground in ROI
            roi_bg = apply_mask(roi, mask_inv)
            # Add projection to ROI background
            dst = cv2.add(roi_bg, projection)
            # Put ROI on SCREEN, in place
            screen[y : y + p_height, x : x + p_width] = dst
        #    self._add_projection(projection, blend=self.pos['m_blend'])
            screen = rotate(screen, angle, self.scr_centre_x, self.scr_centre_y)
        return screen

if __name__ == '__main__':
    #ghost = Ghost('/home/chip/pythoncourse/hologram2/test.mp4')
    ghost = Ghost()

    #path = getcwd()
    #ghost.set_output(path+'/out.avi')

    ghost.run()
