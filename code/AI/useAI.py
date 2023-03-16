import cv2 as cv
import argparse

from Code.image.imageCapture import Cam

class AI:
  def loadAI(self):
    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    parser.add_argument('--faceCascade', help='Path to face cascade.', default='code\image\haarcascade_frontalface_alt.xml')
    parser.add_argument('--eyesCascade', help='Path to eyes cascade.', default='code\image\haarcascade_eye_tree_eyeglasses.xml')
    parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
    args = parser.parse_args()
    
    self.faceCascadeName = args.faceCascade
    self.eyesCascadeName = args.eyesCascade
    self.cameraDevice = args.camera
    self.camera = Cam(self.cameraDevice)
    self.faceCascade = cv.CascadeClassifier()
    self.eyesCascade = cv.CascadeClassifier()
    
    #-- 1. Load the cascades
    if not self.faceCascade.load(cv.samples.findFile(self.faceCascadeName)):
      print('--(!)Error loading face cascade')
      exit(0)
    if not self.eyesCascade.load(cv.samples.findFile(self.eyesCascadeName)):
      print('--(!)Error loading eyes cascade')
      exit(0)
    self.detectAndDisplayFace()

  def detectAndDisplayFace(self):
    while(True):
      frame = self.cameraDevice.readCam()
      frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
      frameGray = cv.equalizeHist(frameGray)
      # Detect faces
      faces = self.faceCascade.detectMultiScale(frameGray)
      print("Found {0} faces!".format(len(faces)))
      for (x,y,w,h) in faces:
        frame = cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        faceROI = frameGray[y:y+h,x:x+w]
        # In each face, detect eyes... not really  needed, but it's cool
        eyes = self.eyesCascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
          eyeCenter = (x + x2 + w2//2, y + y2 + h2//2)
          radius = int(round((w2 + h2)*0.25))
          frame = cv.circle(frame, eyeCenter, radius, (255, 0, 0 ), 4)
      cv.imshow('Capture - Face detection', frame)