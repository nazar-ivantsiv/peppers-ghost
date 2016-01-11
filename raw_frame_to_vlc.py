from __future__ import division
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

sout=#transcode{vcodec=h264,vb=800,scale=Auto,acodec=none}:http{mux=ts,dst=:8080/}

"""

VLC_BIN = 'cvlc'
VIDEO_CODEC = 'mp4v'
VIDEO_BITRATE = 800
HTTP_IP = '192.168.0.2' #'127.0.0.1'
HTTP_PORT = 8080
command_vlc = [VLC_BIN,
        '-',
        '-v',
        '--demux=rawvideo',
        '--rawvid-fps={}'.format(int(round(cap.fps))),
        '--rawvid-width={}'.format(cap.width),
        '--rawvid-height={}'.format(cap.height),
        '--rawvid-chroma=RV24',
        '--sout', 
#        '#std{access=http,mux=avi,dst=192.168.0.2,port=8080}',
        '#transcode{vcodec=%s,vb=%s,width=%s,height=%s,acodec=none}:'
        'std{access=http,mux=ts,dst=%s,port=%s}' % \
        (VIDEO_CODEC, VIDEO_BITRATE, cap.width, cap.height, HTTP_IP, HTTP_PORT),

        'vlc://quit'    # close vlc when finished
        ]

#command_vlc = command_vlc[:8]  # show raw video, rtreived in vlc

#print(command_vlc[9:])

#sp.call(['mkfifo', 'video_pipe'])
pipe = sp.Popen( command_vlc , stdin=sp.PIPE, stderr=sp.STDOUT)

start = time.time()

def add_fps(frame, start):
    fps = c / (time.time() - start)
    cv2.putText(frame, 'fps: {0:.1f}  frame #:{1}'.format(fps, c), (10, cap.height - 30), \
                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2, cv2.LINE_AA)
    return frame

while c < cap.video_len:
    frame = cap.next_frame()
    add_fps(frame, start)
    rgb = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB )
    pipe.stdin.write( rgb.tostring() )

    # PREVIEW
    cv2.imshow('show', frame)
    key = 0xFF & cv2.waitKey(5)
    if key == ord('q'):
        break
    c += 1

cap.release()
pipe.terminate()