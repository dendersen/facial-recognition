from SRC.image.imageCapture import Cam
import cv2 as cv

Camera = Cam(0)
while True:
  pic = Camera.readCam()  
  if cv.waitKey(10) == 32:
    Camera.processFace(pic)
  if cv.waitKey(10) == 27:
    break

# import cv2 as cv

# models = tensorflowModels()

# a:AI = AI()

# while True:
#   a.detectAndDisplayFace()
#   if cv.waitKey(10) == 27:
#     break

