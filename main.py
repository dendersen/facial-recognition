from typing import List
from SRC.image.imageCapture import Camera
import numpy as np
import SRC.image.imageSaver as IS
import cv2 as cv
from time import time

def getValidLabel(msg:str = "please choose a label: ")->str:
  while(True):
    label = input(msg).capitalize()
    if(["Christoffer","David","Niels","Other","Temp"].__contains__(label)):
      return label
    print("not a valid label")
    print("valid labels",end=": ")
    print(["Christoffer","David","Niels","Other"])

faces:List[List[List[List[int]]]]=None; camera=None; label:str=None; waitTime:int=None; overide:bool=None; pic:List[List[List[int]]]=None; Time:int=None; captureTime:int=None; saveAmount:int=None; useTimer:bool=None

def init():
  global faces, camera, label, waitTime, overide, pic, Time, captureTime, saveAmount, useTimer
  # make an instance of camera
  camera = Camera(0)
  faces = []
  label = getValidLabel()
  waitTime = 1 #smaller alows more pictures, higher makes keybord less responsive
  overide = True
  pic = []
  useTimer = True
  captureTime = 5000
  saveAmount = 100
  Time = int(time()*1000)

def run():
  global faces, camera, label, waitTime, overide, pic, Time, captureTime, saveAmount
  while True:
    while True:
      pic = camera.readCam(True)
      if overide or cv.waitKey(waitTime) == 32:
        BGRface = camera.processFace(pic,False,False)
        if type(BGRface) == np.ndarray:
          # save original face
          faces.append(cv.cvtColor(BGRface, cv.COLOR_BGR2RGB))
      if cv.waitKey(waitTime) == 27:
        IS.saveImage(faces,label,False)
        faces = []
        return
      if timer(captureTime):
        IS.saveImage(faces,label,False)
        faces = []
        break
      if(len(faces) >= saveAmount):
        IS.saveImage(faces,label,False)
        faces = []
    if cv.waitKey(captureTime) == 27:
      return
    Time = int(time()*1000)

def timer(mil:int) -> bool:
  global useTimer
  return useTimer and int(time()*1000)-Time > mil

init()
run()