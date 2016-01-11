from time import time

import numpy as np
import cv2

class Output(object):
    """ """
    def __init__(self):
        self._out = None
        self.fps = 20.0
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')

    def set_output(self, file_path, scr_height, scr_width):
        """Define the codec and create VideoWriter object to output video.
        Args:
            file_path -- fila path
            scr_height -- 
            scr_width --
        Note:
            Heigh and width are flipped in this case (width, height).
            Not common to cv2.
        """
        self.scr_width = scr_width
        self.scr_height = scr_height
        self._out = cv2.VideoWriter(file_path, self.fourcc, self.fps, \
                            (scr_width, scr_height))

    @property
    def is_opened(self):
        if self._out != None:
            return self._out.isOpened()
        return False

    def write(self, screen):
        dim = screen.shape[:2]
        if (self.scr_height, self.scr_width) == dim:
            self._out.write(screen)
        else:
            print('Output image size changed. Stop writing to file.')
            self.release()

    def release(self):
        """Finish writing video."""
        if self._out != None:
            self._out.release()