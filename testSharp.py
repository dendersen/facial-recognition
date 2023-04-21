import cv2 as cv
from SRC.image.imageCapture import Camera
from SRC.image.imageEditor import linearSharpen
Cam = Camera(0)

while(True):
  while cv.waitKey(10) != 32:
    pic = Cam.readCam(False)
    cv.imshow("Cam output: ", pic)
  if(type(pic) != type(None)):
    pic = Cam.processFace(pic,info=False)
    if(type(pic) != type(None)):
      
      cv.imshow("Cam output: ", pic)
      linearSharpen(pic)
    else:
      continue
  else:
    continue
  while cv.waitKey(10) != 27: pass