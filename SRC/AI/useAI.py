import cv2 as cv
import argparse

# tjek efter om alle imports skal bruges
import tensorflow as tf
import keras
import keras_vggface
from keras_vggface.vggface import VGGFace
import numpy as np
from keras.utils.data_utils import get_file
import keras_vggface.utils
import PIL
import os
import os.path

from SRC.image.imageCapture import Cam

class tensorflowModels:
  def __init__(self) -> None:
    # self.vggface = VGGFace(model='vgg16')
    # self.vggfaceResnet = VGGFace(model='resnet50')
    # self.vggfaceSenet = VGGFace(model='senet50')
    
    # # Print downloaded weights, input, and output
    # print(self.vggface.summary())
    # print('Inputs are: ', self.vggface.inputs)
    # print('Outputs are: ', self.vggface.outputs)
    pass

class AI:
  def __init__(self):
    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    parser.add_argument('--faceCascade', help='Path to face cascade.', default='SRC\image\haarcascade_frontalface_alt.xml')
    parser.add_argument('--eyesCascade', help='Path to eyes cascade.', default='SRC\image\haarcascade_eye_tree_eyeglasses.xml')
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
  
  def detectAndDisplayFace(self):
    frame = self.camera.readCam()
    frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frameGray = cv.equalizeHist(frameGray)
    # Detect faces
    faces = self.faceCascade.detectMultiScale(frameGray)
    print("Found {0} faces!".format(len(faces)))
    for (faceX,faceY,faceWidth,faceHeight) in faces:
      frame = cv.rectangle(frame, (faceX, faceY), (faceX+faceWidth, faceY+faceHeight), (0, 255, 0), 2)
      faceROI = frameGray[faceY:faceY+faceHeight,faceX:faceX+faceWidth]
      # In each face, detect eyes... not really  needed, but it's cool
      eyes = self.eyesCascade.detectMultiScale(faceROI)
      for (x2,y2,w2,h2) in eyes:
        eyeCenter = (faceX + x2 + w2//2, faceY + y2 + h2//2)
        radius = int(round((w2 + h2)*0.25))
        frame = cv.circle(frame, eyeCenter, radius, (255, 0, 0 ), 4)
    cv.imshow('Capture - Face detection', frame)