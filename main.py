from SRC.image.imageCapture import Cam
import cv2 as cv

# make an instance of camera
Camera = Cam(0)
isHeldDown:bool = False

while True:
  pic = Camera.readCam()  
  if cv.waitKey(10) == 32:
    isHeldDown = True
    face = Camera.processFace(pic)
    
  if cv.waitKey(10) == 27:
    break

# import cv2 as cv

# models = tensorflowModels()

# a:AI = AI()

# while True:
#   a.detectAndDisplayFace()
#   if cv.waitKey(10) == 27:
#     break
