import cv2 as cv
import mtcnn
import numpy as np
class Cam:
  def __init__(self,cameraDevice:int) -> None:
    self.cameraDevice = cv.VideoCapture(cameraDevice)
  
  def readCam(self)->list[list[int]]:
    #-- 2. Read the video stream
    if not self.cameraDevice.isOpened:
      print('--(!)Error opening video capture')
      exit(0)
    ret, frame = self.cameraDevice.read()
    frame:list[list[int]]
    if frame is None:
      print('--(!) No captured frame -- Break!')
      return
    cv.imshow('Cam output: ', frame)
    return frame
  
  def processFace(self, frame):
    # Makes the face detector
    self.faceDetector = mtcnn.MTCNN()
    faceInformation = self.faceDetector.detect_faces(frame)
    if faceInformation != []:
      boundingBox = faceInformation[0]['box']
      x, y, width, height = boundingBox[0], boundingBox[1], boundingBox[2], boundingBox[3]
      cv.rectangle(frame, 
                    (x, y),
                    (x+width,y+height),
                    (0,155,255),
                    2)
      
      face = frame[y:y+height, x:x+width]
      cv.imshow('This is the face', face)
      return face
    
    # x, y, width, height = faceInformation[0]['box']
    
    #display resulting frame