import cv2 as cv
from facenet_pytorch.models.mtcnn import MTCNN
import numpy as np
import torch
from typing import List

class Camera:
  def __init__(self,cameraDevice:int) -> None:
    self.cameraDevice = cv.VideoCapture(cameraDevice)
    # Makes the face detector
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.mtcnn = MTCNN(min_face_size=120, select_largest=True, device=device)
  
  def readCam(self,show:bool = True)->List[List[int]]:
    #-- 2. Read the video stream
    if not self.cameraDevice.isOpened:
      print('--(!)Error opening video capture')
      exit(0)
    ret, frame = self.cameraDevice.read()
    frame:list[list[int]]
    if frame is None:
      print('--(!) No captured frame -- Break!')
      return
    if(show):
      cv.imshow('Cam output: ', frame)
    return frame
  
  def close(self):
    self.cameraDevice.release()
  
  def processFace(self, frame,info:bool = True,show:bool = False) -> List[List[List[int]]]:
    # get fram shape
    height, width, channel = frame.shape
    
    # prdict face
    face, probs = self.mtcnn.detect(frame)
    
    if type(face) != np.ndarray:
      if info:
        print("there is no face!")
    else:
      xLeft = int(min(face[0][0], face[0][2]))
      xRight = int(max(face[0][0], face[0][2]))
      yBottom = int(min(face[0][1], face[0][3]))
      yTop = int(max(face[0][1], face[0][3]))
      
      faceWidth = xRight-xLeft
      faceHeight = yTop-yBottom
      
      total = max(faceWidth,faceHeight)
      
      xdif = (total - faceWidth)/2
      ydif = (total - faceHeight)/2
      
      xLeft = int((xLeft-xdif)-(total/4))
      xRight = int((xRight+xdif)+(total/4))
      yBottom = int((yBottom-ydif)-(total/4))
      yTop = int((yTop+ydif)+(total/4))
      
      if (xLeft < 0 or xRight > width or yBottom < 0 or yTop > height):
        if info:
          print('ERROR! Face not in frame, please move to center')
      else:
        buff2 = frame[yBottom:yTop, xLeft:xRight]
        if(info):
          print("We found that there is: " + str(probs[0]) + "% that it is a face")
        if(show):
          cv.imshow('This is the face', buff2)
        print("We found that there is: " + str(probs[0]) + "% that it is a face")
        cv.imshow('This is the face', buff2)
        return buff2
