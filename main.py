from SRC.image.imageCapture import Camera
import numpy as np
import SRC.image.imageSaver as IS
import cv2 as cv

# make an instance of camera
Camera = Camera(0)
faces = []
label = "David" #chooses label

while True:
  pic = Camera.readCam()
  if cv.waitKey(10) == 32:
    BGRface = Camera.processFace(pic,False)
    if type(BGRface) == np.ndarray:
        # save original face
        faces.append(cv.cvtColor(BGRface, cv.COLOR_BGR2RGB))
  if(len(faces) > 100):
    IS.saveImage(faces,label,False)
    faces = []
  if cv.waitKey(10) == 27:
    break

IS.saveImage(faces,label,False)
