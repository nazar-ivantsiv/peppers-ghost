from threading import Thread
import logging

from . import cv2
from . import np


class Capture(object):
    """Camera or video file input routines in the separate thread.
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
            self.cur_frame_num = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.stopped = False                    # If thread should be stopped
        self.grab ,self.frame = self._cap.read()
        # Create thread
        Thread(target=self.update, args=()).start() 

    def start(self):
        """Start the thread to read frames from the video stream"""
        if self.stopped:
            self.stopped = not self.stopped
            Thread(target=self.update, args=()).start() 
            return self
 
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                break
            # otherwise, GRAB the next frame from the stream
            self.grab = self._cap.grab()
  
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def is_opened(self):
        return not self.stopped

    def next_frame(self):
        """Returna the frame most recently read. 
        Updates cur_frame_num count.
        Loops video if _loop_video = True
        """
        # Get cur frame num
        self.cur_frame_num = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        if self.source < 0:
            # source is video file
            if self.cur_frame_num == self.video_len:
                # reached the end of file -> goto first frame
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # retrieve recent frame from cap
        if self.grab:
            _, self.frame = self._cap.retrieve()
        return self.frame

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
            self.stop()
            self._cap.release()
    