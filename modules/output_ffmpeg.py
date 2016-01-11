from . import np
from . import cv2
import subprocess as sp

FFMPEG_BIN = "ffmpeg" # on Linux ans Mac OS
#FFMPEG_BIN = "ffmpeg.exe" # on Windows
#VLC_BIN = 'vlc'


class Output(object):
    """
Example:
ffmpeg -f rawvideo -pix_fmt bgr24 -s 640x480 -r 30 -i - -an -f avi -r 30 foo.avi
"""
    def __init__(self):
        self._pipe = None
        self._is_opened = False
        #self.fps = 20.0
        #self.fourcc = cv2.VideoWriter_fourcc(*'XVID')

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
        command = [ FFMPEG_BIN,
        '-f', 'mpegts "tcp://127.0.0.1:2000"',
        '-vcodec','rawvideo',
        '-s', '{}x{}'.format(scr_width, scr_height), # size of one frame
        '-pix_fmt', 'rgb24',
        '-re'
        '-i', '-', # The imput comes from a pipe
        '-an', # Tells FFMPEG not to expect any audio
        '-vcodec', 'mpeg',
        file_path ]

        #command = [ FFMPEG_BIN,
        #'-y', # (optional) overwrite output file if it exists
        #'-f', 'rawvideo',
        #'-vcodec','rawvideo',
        #'-s', '{}x{}'.format(scr_width, scr_height), # size of one frame
        #'-pix_fmt', 'rgb24',
        #'-r', '24', # frames per second
        #'-i', '-', # The imput comes from a pipe
        #'-an', # Tells FFMPEG not to expect any audio
        #'-vcodec', 'mpeg',
        #file_path ]
        self._pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.PIPE)
        self._is_opened = True

    @property
    def is_opened(self):
        return self._is_opened


    def write(self, screen):
        self._pipe.proc.stdin.write(screen.tostring())


    def release(self):
        """Finish writing video."""
        #if self._out != None:
        #    self._out.release()
        self._is_opened = False