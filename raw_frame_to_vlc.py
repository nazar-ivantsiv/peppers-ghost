from __future__ import division
import sys
import time

import numpy as numpy
import cv2
import subprocess as sp

from modules.capture import Capture

"""Use this way, from shell:
python raw_frame_to_vlc.py | cvlc - --sout '#standard{access=http,mux=ts,url=127.0.0.1:8080}'
"""

cap = Capture('test.mp4')
c = 0

FFMPEG_BIN = "ffmpeg" # on Linux ans Mac OS
command = [ FFMPEG_BIN,
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', '{}x{}'.format(cap.width, cap.height), # size of one frame
        '-pix_fmt', 'rgb24',
        '-r', '24', # frames per second
        '-i', '-', # The imput comes from a pipe
        '-an', # Tells FFMPEG not to expect any audio
        '-f', 'mpeg',
        '-' # output goes to stdout (use 'vlc -' to catch)
        ]


#VLC_BIN = 'cvlc'
#command_vlc = [VLC_BIN,
#        '-',
#        '--sout', '\'#standard{access=http,mux=ts,url=127.0.0.1:8080}\'',
#        '--play-and-exit'
#            ]

pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.PIPE)
#pipe_vlc = sp.Popen( command_vlc, stdout=sp.PIPE, stderr=sp.PIPE)

start = time.time()

def add_fps(frame, start):
    fps = c / (time.time() - start)
    cv2.putText(frame, 'fps: {0:.2f}  frame #: {1}'.format(fps, c), (10, cap.height - 50), \
                cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 2, cv2.LINE_AA)
    return frame


while c < cap.video_len:
    frame = cap.next_frame()
    add_fps(frame, start)
    #print('frame #:{}'.format(c))
    rgb = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB )
    #sys.stdout.write( rgb.tostring() )
    pipe.stdin.write( rgb.tostring() )

    # - DONE. pass array.tostring() to ffmpeg
    # - pass array.tostring() to vlc -I rc -vvv - --sout '#standard{access=http,mux=ts,url=127.0.0.1:8080}'
    # - try to send array.tostring() via socket and read with vlc
    # - OR send encoded MPEG frames via socket and read with vlc

    # PREVIEW
#    cv2.imshow('show', frame)
#    key = 0xFF & cv2.waitKey(1)
#    if key == ord('q'):
#        break
    c += 1

pipe.stdin.write('^C')
cap.release()
pipe.terminate()
#pipe2.terminate()