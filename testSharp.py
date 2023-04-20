from SRC.image.imageEditor import sharpen
from SRC.image.imageCapture import Camera
import cv2 as cv

cam = Camera(0)
while(not cv.waitKey(10) == 32):
  pic = cam.readCam()

pic = sharpen(pic,threshold=2,showSteps=True,strength=0.9,amplification = 1.03)#keep amplification as low as possible!! or the original image will shine through, can be mitigated with threshold

cv.imshow('sharp output: ',pic)
while(not cv.waitKey(10) == 27):pass;