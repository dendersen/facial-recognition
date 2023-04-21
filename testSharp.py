import cv2 as cv
from SRC.image.imageCapture import Camera
from SRC.image.imageEditor import sharpen
Cam = Camera(0)

while(True):
  while cv.waitKey(10) != 32:
    pic = Cam.readCam(False)
    cv.imshow("Cam output: ", pic)
  if(type(pic) != type(None)):
    pic = Cam.processFace(pic,info=False)
    if(type(pic) != type(None)):
      cv.imshow("Cam output: ", pic)
      sharpen(pic,strength=0.3,threshold=50,showSteps=True,showEnd=True,amplification=1)
    else:
      continue
  else:
    continue
  while cv.waitKey(10) != 27: pass