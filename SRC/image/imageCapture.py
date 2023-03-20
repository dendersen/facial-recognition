import cv2 as cv
from facenet_pytorch.models.mtcnn import MTCNN
import numpy as np
import torch

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
  
  def processFace(self, frame) -> list[list[list[int]]]:
    # get fram shape
    height, width, channel = frame.shape
    
    # Makes the face detector
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(min_face_size=120, select_largest=True, device=device)
    # prdict face
    face, probs = mtcnn.detect(frame)
    
    if type(face) != np.ndarray:
      print("there is no face!")
    else:
      xLeft = int(min(face[0][0], face[0][2]))
      xRight = int(max(face[0][0], face[0][2]))
      yBottom = int(min(face[0][1], face[0][3]))
      yTop = int(max(face[0][1], face[0][3]))
      
      # not needed
      # cv.rectangle(frame, (xLeft, yLeft), (xRight, yRight), 
      #                       (255, 0, 0), 2)
      
      faceWidth = xRight-xLeft
      faceHeight = yTop-yBottom
      
      total = max(faceWidth,faceHeight)
      
      xdif = (total - faceWidth)/2
      ydif = (total - faceHeight)/2
      
      xLeft = int((xLeft-xdif)-(total/4))
      xRight = int((xRight+xdif)+(total/4))
      yBottom = int((yBottom-ydif)-(total/4))
      yTop = int((yTop+ydif)+(total/4))
      
      if xLeft < 0 or xRight > width or yBottom < 0 or yTop > height:
        print('ERROR! Face not in frame, please move to center')
      else:
        buff2 = frame[yBottom:yTop, xLeft:xRight]
        print("We found that there is: " + str(probs[0]) + "% that it is a face")
        cv.imshow('This is the face', buff2)
        return buff2