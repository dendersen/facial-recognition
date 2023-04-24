import cv2 as cv
from SRC.image.imageCapture import Camera
from SRC.image.imageEditor import makeVarients
import numpy as np
Cam = Camera(0)

def run():
  global pic
  while True:
    pic = Cam.readCam(False)
    if type(pic) != type(None):
      pic = Cam.processFace(pic)
      if(type(pic) != type(None)):
        cv.imshow("in", pic)
        cv.imshow("out",np.array(makeVarients(pic,1)[0]))
      else:
        continue
    else:
      continue
    if cv.waitKey(200) == 27:
      exit()

run()
