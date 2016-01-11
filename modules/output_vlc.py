from __future__ import division
from time import time

from . import np
from . import cv2
import subprocess as sp

VLC_BIN = 'cvlc'
FILE_CACHING = 300      # 300
NETWORK_CACHING = 1000   # 1000

VIDEO_CODEC = 'mp4v'    # mp4v
VIDEO_BITRATE = 1600     # 800

MUX = 'ts'
DST = '192.168.0.2:8080' #'127.0.0.1:8080'



class Output(object):
    """ """
    def __init__(self):
        self._pipe = None
        self._is_opened = False

    def set_output(self, fps=None, scr_height=None, scr_width=None):
        """Define the codec and create VideoWriter object to output video.
        Args:
            file_path -- not used in VLC streaming
            scr_height -- 
            scr_width --
        Note:
            Heigh and width are flipped in this case (width, height).
            Not common to OpenCV.
        """
        print('\nStarting stream to {}:'.format(DST))
        print('FPS rate:   {}'.format(fps))
        print('resolution: {} x {}'.format(scr_width, scr_height))
        command_vlc = [VLC_BIN,
'-',
'-v',
'--file-caching={}'.format(FILE_CACHING),
'--network-caching={}'.format(NETWORK_CACHING),
'--clock-synchro=0',
'--demux=rawvideo',
'--rawvid-fps={}'.format(int(round(fps))),
'--rawvid-width={}'.format(scr_width),
'--rawvid-height={}'.format(scr_height),
'--rawvid-chroma=RV24',
'--sout', 
'#transcode{vcodec=%s,vb=%s,width=%s,height=%s,acodec=none}:'
'std{access=http,mux=%s,dst=%s}' % \
(VIDEO_CODEC, VIDEO_BITRATE, scr_width, scr_height, MUX, DST),
'vlc://quit'    # close vlc when finished
        ]

        self._pipe = sp.Popen( command_vlc , stdin=sp.PIPE, stderr=sp.STDOUT)
        self._is_opened = True

    @property
    def is_opened(self):
        return self._is_opened


    def write(self, screen):
        #rgb = cv2.cvtColor( screen, cv2.COLOR_BGR2RGB )
        self._pipe.stdin.write( screen.tostring() )


    def release(self):
        """Finish writing video."""
        if self._pipe != None:
            self._pipe.terminate()
            self._is_opened = False