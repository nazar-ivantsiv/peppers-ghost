from __future__ import division
import sys
import time

import numpy as numpy
import cv2
import subprocess as sp

from modules.capture import Capture

"""Use this way, from shell:
python raw_frame_to_vlc.py | cvlc -vvv --demux=rawvideo --rawvid-fps=24 
--rawvid-width=640 --rawvid-height=360 --rawvid-chroma=RV24 - 
--sout "#transcode{vcodec=h264,vb=800,width=640, height=360}:standard{access=file,mux=ts,dst=test.mpg}" vlc://quit
"""

cap = Capture('test2.mp4')
c = 0
"""
FFMPEG_BIN = "ffmpeg" # on Linux ans Mac OS
command = [ FFMPEG_BIN,
        '-y'
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', '{}x{}'.format(cap.width, cap.height), # size of one frame
        '-pix_fmt', 'rgb24',
        '-r', '24', # frames per second
        '-i', '-', # The imput comes from a pipe
        '-an', # Tells FFMPEG not to expect any audio
        '-f', 'mpeg',
        '-'# output goes to stdout (use 'vlc -' to catch)
        ]
"""


VLC_BIN = 'cvlc'
command_vlc = [VLC_BIN,
        '-vvv',
        '--demux=rawvideo',
        '--rawvid-fps=24',  # Use cap value
        '--rawvid-width={}'.format(cap.width),
        '--rawvid-height={}'.format(cap.height),
        '--rawvid-chroma=RV24',
        '-',
        '--sout', 
        '\'#transcode{vcodec=mp4v,vb=800,width=640, height=360}:std{access=http,mux=ts,dst=127.0.0.1,port=8080}\'',
        'vlc://quit'
        ]
#sp.call(['mkfifo', 'video_pipe'])
#pipe = sp.Popen( command_vlc , stdin=sp.PIPE, stderr=sp.PIPE)

start = time.time()

def add_fps(frame, start):
    fps = c / (time.time() - start)
    cv2.putText(frame, 'fps: {0:.1f}  frame #:{1}'.format(fps, c), (10, cap.height - 30), \
                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2, cv2.LINE_AA)
    return frame


while c < cap.video_len:
    frame = cap.next_frame()
    add_fps(frame, start)
    #print('frame #:{}'.format(c))
    rgb = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB )
    sys.stdout.write( rgb.tostring() )
    #pipe.stdin.write( rgb.tostring() )

    # - DONE. pass array.tostring() to ffmpeg
    # - pass array.tostring() to vlc -I rc -vvv - --sout '#standard{access=http,mux=ts,url=127.0.0.1:8080}'

    # PREVIEW
    cv2.imshow('show', frame)
    key = 0xFF & cv2.waitKey(5)
    if key == ord('q'):
        break
    c += 1

#pipe.stdin.write('^C')
cap.release()
#pipe.terminate()