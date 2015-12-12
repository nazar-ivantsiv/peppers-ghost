#!bin/python
from os import getcwd

from ghost import Ghost

#ghost = Ghost('/home/chip/pythoncourse/hologram2/test.mp4')
ghost = Ghost()

#path = getcwd() + '/out.avi'
#ghost.set_output(path)

ghost.run()


