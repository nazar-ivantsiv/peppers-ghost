from __future__ import division
from . import cv2
from . import np


class Gui(object):
    """GUI for Pepper's Ghost video processor.
    Args:
        height -- output SCREEN original height
        width -- output SCREEN original width
        pos -- reference to dict with default trackbar values (from ghost.py)
    Returns:
        self.pos['scr_width'] -- screen width
        self.pos['m_cntr'] -- mask vertical position on frame (d. MASK_CENTRE)
        self.pos['m_btm'] -- mask bottom edge position (def. MASK_BOTTOM)
        self.pos['m_side'] -- distance between mask side corners (d. MASK_SIDE)
        self.pos['m_blend'] -- blends 4 mask projections (def. MASK_BLEND)
        self.pos['i_x'] -- frame x pos
        self.pos['i_y'] -- frame y pos
        self.pos['scale'] -- frame scale factor
        self.pos['angle'] -- angle relation between projections
        self.pos['projections'] -- projections qty (def. 4)
        self.pos['loop_video'] -- on/off
        self.pos['tracking_on'] -- FaceTracking on/off
        self.pos['gc_iters'] -- GrabCut iterations
        self.pos['contrast'] -- contrast adj.
        self.pos['brightness'] -- brightness adj.
    """
    DEBUGGER_MODE = 0                           # Flag: debugger mode off
    C_HDR = 'SETTINGS'                          # Controls window header
    S_HDR = 'SCREEN'                            # Screen window header

    def __init__(self, height, width, pos):
        self.height = height
        self.width = width
        self.pos = pos                          # Trackbar positons dict
        self.fullscreen = not cv2.WINDOW_FULLSCREEN # Flag: fullscreen (off)
        self._debugger_off = not self.DEBUGGER_MODE # Flag: debugger status

        def on_change(_):
            """Updates values in self.pos if any change."""
            self._get_trackbar_values()
        
        cv2.namedWindow(self.C_HDR, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.S_HDR, cv2.WINDOW_NORMAL)
#        cv2.createTrackbar('fit width', self.C_HDR, int(self.width * 2),\
#                           self.width * 4, on_change)
        cv2.createTrackbar('mask centre', self.C_HDR, int(self.pos['m_cntr'] * 100),\
                           100, on_change)
        cv2.createTrackbar('mask bottom', self.C_HDR, int(self.pos['m_btm'] * 100),\
                           100, on_change)
        cv2.createTrackbar('mask side', self.C_HDR, int(self.pos['m_side'] * 100), 400, on_change)
        cv2.createTrackbar('mask blend', self.C_HDR, int(self.pos['m_blend'] * 1000),\
                           1000, on_change)
        cv2.createTrackbar('image x', self.C_HDR, int(self.width / 2), \
                           self.width, on_change)
        cv2.createTrackbar('image y', self.C_HDR, int(self.height / 2), \
                           self.height, on_change)
        cv2.createTrackbar('scale', self.C_HDR, int(self.pos['scale'] * 100), 300, on_change)
        cv2.createTrackbar('angle', self.C_HDR, self.pos['angle'], 90, on_change)
        cv2.createTrackbar('projections', self.C_HDR, self.pos['projections'], 10, on_change)
        cv2.createTrackbar('loop video', self.C_HDR, self.pos['loop_video'], 1, on_change)
        cv2.createTrackbar('Track Faces', self.C_HDR, self.pos['tracking_on'], 1, on_change)
        cv2.createTrackbar('  GrabCut iters', self.C_HDR, self.pos['gc_iters'], 5, on_change)
        cv2.createTrackbar('contrast', self.C_HDR, int(self.pos['contrast'] * 10), 30, on_change)
        cv2.createTrackbar('brightness', self.C_HDR, self.pos['brightness'], 100, on_change)
        self._get_trackbar_values()

    def _get_trackbar_values(self):
        """Refreshes variables with Trackbars positions.
        on_changes: 
            self.pos -- positions, states and coeficients dict
        """
#        self.pos['scr_width'] = \
#            max((cv2.getTrackbarPos('fit width', self.C_HDR) // 2) * 2 , \
#                self.height * 2)
        self.pos['m_cntr'] = cv2.getTrackbarPos('mask centre', self.C_HDR) / 100
        self.pos['m_btm'] = cv2.getTrackbarPos('mask bottom', self.C_HDR)  / 100
        self.pos['m_side'] = cv2.getTrackbarPos('mask side', self.C_HDR) / 100
        self.pos['m_blend'] = cv2.getTrackbarPos('mask blend', self.C_HDR) / 1000
        self.pos['i_x'] = cv2.getTrackbarPos('image x', self.C_HDR) - \
                                                self.width / 2
        self.pos['i_y'] = cv2.getTrackbarPos('image y', self.C_HDR) - \
                                                self.height / 2
        self.pos['scale'] = cv2.getTrackbarPos('scale', self.C_HDR) / 100 or 0.01
        self.pos['angle'] = -cv2.getTrackbarPos('angle', self.C_HDR)
        self.pos['projections'] = cv2.getTrackbarPos('projections', self.C_HDR)
        self.pos['loop_video'] = cv2.getTrackbarPos('loop video', self.C_HDR)
        self.pos['tracking_on'] = cv2.getTrackbarPos('Track Faces', self.C_HDR)
        self.pos['gc_iters'] = cv2.getTrackbarPos('  GrabCut iters', self.C_HDR)
        self.pos['contrast'] = cv2.getTrackbarPos('contrast', self.C_HDR) / 10
        self.pos['brightness'] = cv2.getTrackbarPos('brightness', self.C_HDR)

    def preview(self, screen):
        cv2.imshow(self.S_HDR, screen)

    def toggle_fullscreen(self):
        """Switches SCREEN window fullscreen mode on/off"""
        self.fullscreen = not self.fullscreen
        cv2.setWindowProperty(self.S_HDR, cv2.WND_PROP_FULLSCREEN, \
                              self.fullscreen)

    def toggle_debugger(self):
        """Switches DEBUGGER windows on/off"""
        self.DEBUGGER_MODE = not self. DEBUGGER_MODE
        if self.DEBUGGER_MODE:
            self._debugger_off = False
        else:
            if not self._debugger_off:
                cv2.destroyWindow('original')
                cv2.destroyWindow('result')
                self._debugger_off = True

    def debugger_show(self, frame, frame_mod):
        """Adds two additional windows for debugging and adjustments.
        Args:
            frame -- original frame from video input, with face highlited
            frame_mod -- processed frame
        """
        cv2.imshow('original', frame)
        cv2.moveWindow('original', 0, 0)
        cv2.imshow('result', frame_mod)
        cv2.moveWindow('result', 0, self.height + 50)

    def exit(self):
        cv2.destroyAllWindows()