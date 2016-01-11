from . import cv2
from . import np


class Capture(object):
    """Camera or video file input routines.
    Args:
        source -- video source (0, 1, 2.. - camera or video file path)
    """
    def __init__(self, source=0):
        self._cap = cv2.VideoCapture(source)    # Input source instance
        if not self._cap.isOpened():
            self._cap.open()
        self.source = source                    # ''>= 0'-camera, '-1' - file
        if isinstance(source, str):
            # Source is string == path to video file
            self.source = -1
            self._loop_video = True
            self.video_len = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.cur_frame = self._cap.get(cv2.CAP_PROP_POS_FRAMES)

    def is_opened(self):
        return self._cap.isOpened()

    def next_frame(self):
        """Captures frame from source. 
        Updates cur_frame count.
        Loops video if _loop_video = True
        """
        self.cur_frame = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        if self.source < 0:
            # video file
            if self.cur_frame == self.video_len:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self._cap.read()
        return frame

    @property
    def loop_video(self):
        """Sets video loop play status on/off"""
        return self._loop_video

    @loop_video.setter
    def loop_video(self, status):
        if isinstance(status, bool):
            self._loop_video = status
        elif isinstance(status, int):
            self._loop_video = [False, True][status]
        else:
            raise TypeError

    @property
    def height(self):
        return int(self._cap.get(4))

    @property
    def width(self):
        return int(self._cap.get(3))

    @property
    def fps(self):
        return self._cap.get(cv2.CAP_PROP_FPS)

    def release(self):
        if self._cap.isOpened():
            self._cap.release()
    