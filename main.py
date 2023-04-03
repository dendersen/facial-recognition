from SRC.image.imageCapture import Camera
from SRC.image.imageEditor import makeVarients,modifyOriginals
import numpy as np
import SRC.image.imageLoader as IL
import SRC.image.imageSaver as IS
import cv2 as cv

modifyOriginals()

# make an instance of camera
Camera = Camera(0)

while True:
  pic = Camera.readCam()
  if cv.waitKey(10) == 32:
    BGRface = Camera.processFace(pic)
    if type(BGRface) == np.ndarray:
        # save original face
        RGBface = cv.cvtColor(BGRface, cv.COLOR_BGR2RGB)
        print("This is the shape of the face picture: ", RGBface.shape)
        IS.saveImage([RGBface],"Christoffer",False)
  if cv.waitKey(10) == 27:
    break
