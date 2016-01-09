import sys

import numpy as numpy
import cv2
import subprocess as sp

from modules.capture import Capture


cap = Capture('out.avi')
c = 0

FFMPEG_BIN = "ffmpeg" # on Linux ans Mac OS
command = [ FFMPEG_BIN,
        '-y', # (optional) overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', '{}x{}'.format(cap.width, cap.height), # size of one frame
        '-pix_fmt', 'rgb24',
        '-r', '24', # frames per second
        '-i', '-', # The imput comes from a pipe
        '-an', # Tells FFMPEG not to expect any audio
        'my_output_videofile.mp4' ]

pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.PIPE)

while c < cap.video_len:
    frame = cap.next_frame()
    #cv2.imshow('show', frame)

    pipe.stdin.write( frame.tostring() )

    #DONE. pass array.tostring() to ffmpeg
    # pass array.tostring() to vlc -I rc -vvv - --sout '#standard{access=http,mux=ts,url=127.0.0.1:8080}'
    # try to send array.tostring() to socket and read with vlc

    #key = 0xFF & cv2.waitKey(5)
    #if key == ord('q'):
    #    break
    c += 1

cap.release()
pipe.terminate()