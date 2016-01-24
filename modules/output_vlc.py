from __future__ import division
from time import time, sleep
from threading import Thread

from . import np
from . import cv2
import subprocess as sp


VLC_BIN = 'cvlc'
FPS = 25

VIDEO_CODEC = 'mp4v'    # mp4v
VIDEO_BITRATE = 4000     # 4000

MUX = 'ts'
DST = ':8080/'



class Output(object):
    """ """
    def __init__(self, dst=':8080/'):
        self._is_opened = False
        self._pipe = None

    def set_output(self, scr_height=None, scr_width=None):
        """Define the codec and create VideoWriter object to output video.
        Args:
            scr_height -- 
            scr_width --
        Note:
            Heigh and width are flipped in this case (width, height).
            Not common to OpenCV.
        """

        command_vlc = [VLC_BIN,
'-',
'-v',
'--file-caching=1000',      # Check if really required
'--network-caching=1000',   # Check if really required
'--demux=rawvideo',
'--rawvid-fps={}'.format(FPS),
'--rawvid-width={}'.format(scr_width),
'--rawvid-height={}'.format(scr_height),
'--rawvid-chroma=RV24',
'--sout', 

'#transcode{vcodec=%s,vb=%s,width=%s,height=%s,acodec=none}:'
'http{mux=%s,dst=%s}' % \
(VIDEO_CODEC, VIDEO_BITRATE, scr_width, scr_height, MUX, DST),

'--sout-keep',
'vlc://quit'    # close vlc when finished
        ]
        if self._is_opened == False:
            print('\nStarting stream to {}:'.format(DST))
            print('FPS rate:   {}'.format(FPS))
            print('Bitraterate:   {}'.format(VIDEO_BITRATE))
            print('resolution: {} x {}'.format(scr_width, scr_height))
            self._pipe = sp.Popen( command_vlc , stdin=sp.PIPE, stderr=sp.STDOUT)
            self._is_opened = True
            self.output_img = np.zeros((scr_height, scr_width, 3), np.uint8) # Just to initialise
            # Create a thread for output loop
            self._out = Thread(target=self.output, args=()).start() 

    @property
    def is_opened(self):
        return self._is_opened

    def write(self, output_img):
        self.output_img = output_img

    def output(self):
        """This method is launshed in separate thread 'self._out' to send 
        frames to vlc with a stable fps, indeendent from capturing and
        processing performance.
        """
        count = 0
        delta = 1 / FPS
        while self._is_opened:
            start = time()
            # Send the frame to VLC
            self._pipe.stdin.write( self.output_img.tostring() )
            # Calculate time spent to output the frame
            diff = time() - start
            if diff < delta:
                # Wait remaining amount of time to produce correct FPS.
                sleep(delta - diff)
            count += 1
        self._pipe.terminate()

    def release(self):
        """Finish writing video."""
        if self._pipe != None:
            self._is_opened = False