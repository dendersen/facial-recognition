from SRC.image.imageSaver import saveImage
from SRC.image.imageLoader import ProcessOther, loadImgAsArr
from typing import List, Tuple
import tensorflow as tf
import os
from typing import List
import cv2 as cv
import numpy as np
# TODO lav en funktion som:
# tager imod et kvadratisk billede af n- størelse af typen nr array 
# billedet har er en buffer rundt om ansigtet

# Skal rykke billedet rundt, men hele ansigtet skal stadig være i frame 
# der skal laves flere forskælige af disse rykkede ansigter

# for værd af disse nye billeder skal der
# laves naturlige lindende versioner af billedet

# sender disse billeder til image saver


import numpy as np
import random

from SRC.image.imageSaver import saveImage

def makeVarients(image: List[List[List[int]]], variantNumber:int = 10) -> List[List[List[List[int]]]]: 
  # Height and width is always the same. It's defined as size
  size  = image.shape[0]
  
  buffer1Size = int(size * 0.8)
  maxOffset = int(size-buffer1Size)
  
  faces: list[np.ndarray[np.ndarray[np.ndarray[int]]]] = []
  
  for i in range(variantNumber):
    xStart =  random.randint(0,maxOffset)
    xEnd = xStart + buffer1Size
    yStart = random.randint(0,maxOffset)
    yEnd = yStart + buffer1Size
    newVariant = image[yStart:yEnd, xStart:xEnd]
    
    faces.append(newVariant)
  return faces

def clearModified():
  clearPath('images\\modified\\Christoffer')
  clearPath('images\\modified\\Niels')
  clearPath('images\\modified\\David')
  clearPath('images\\modified\\Other')

def clearPath(path:str):
  print('\n Removeing images from: '+ path)
  progbar = tf.keras.utils.Progbar(len(os.listdir(path))-1)
  i = 0
  for file_name in os.listdir(path):
    # construct full file path
    file = os.path.join(path, file_name)
    if ".jpg" in file:
      os.remove(file)
      i = i+1
      progbar.update(i)
    else:
      try:
        clearPath(file)
      except:
        pass

def modifyOriginals(maximum:int = 300,varients:int = 10,dataset:bool = False):
  clearPath('images\\modified\\Christoffer')
  clearPath('images\\modified\\Niels')
  clearPath('images\\modified\\David')
  
  IDChris = 0
  IDDavid = 0
  IDNiels = 0
  IDOther = 0
  
  for image in loadImgAsArr(maximum,False,cropOri = False):
    ID = 0
    if(image[1] == "Christoffer"):
      ID = IDChris
      IDChris += varients
    elif(image[1] == "David"):
      ID = IDDavid
      IDDavid += varients
    elif(image[1] == "Niels"):
      ID = IDNiels
      IDNiels += varients
    else:
      ID = IDOther
      IDOther += varients
    saveImage(makeVarients(image[0],varients),image[1],True,ID,forceID=True,)
  
  if(dataset):
    ProcessOther()

def sharpen(pic:List[List[List[int]]], strength:float = 0.2,threshold = -1, showSteps:bool = False, showEnd:bool = False, amplification:float = 1):
  process = smooth(pic,threshold)
  if(showSteps):
    cv.imshow('smooth output: ',cv.convertScaleAbs(np.array(process)))
  process = difference(pic,process,amplification)
  if(showSteps):
    cv.imshow('detail output: ',cv.convertScaleAbs(np.array(process)))
  pic = combine(pic,process,strength,threshold)
  if(showEnd):
    cv.imshow('sharp output: ',cv.convertScaleAbs(np.array(pic)))
  return pic

def smooth(pic:List[List[List[int]]],threshold = -1,strong:float = 1.0,central:float = 1.0) -> List[List[List[int]]]:
  orgPic = pic.copy()
  tempList = orgPic.copy()
  for y in range(len(pic)-2):
    for x in range(len(pic[y])-2):
      for col in range(len(pic[y][x])):
        tempList[y+1][x+1][col] = int(((
          float(orgPic[y][x][col])/(255.0)*strong+
          float(orgPic[y+1][x][col])/(255.0)*strong+
          float(orgPic[y+2][x][col])/(255.0)*strong+
          float(orgPic[y][x+1][col])/(255.0)*strong+
          float(orgPic[y+1][x+1][col])/(255.0)*central+
          float(orgPic[y+2][x+1][col])/(255.0)*strong+
          float(orgPic[y][x+2][col])/(255.0)*strong+
          float(orgPic[y+1][x+2][col])/(255.0)*strong+
          float(orgPic[y+2][x+2][col])/(255.0)*strong
        )/(central+8*strong))*255)
        if(abs(int(tempList[y][x][col]) - int(orgPic[y][x][col])) < threshold):
          tempList[y][x][col] = orgPic[y][x][col]
  return tempList

def difference(pic1:List[List[List[int]]],pic2:List[List[List[int]]],amplification:float = 1) -> List[List[List[int]]]:
  tempPic1 = cv.cvtColor(pic1, cv.COLOR_BGR2RGB)
  tempPic2 = cv.cvtColor(pic2, cv.COLOR_BGR2RGB)
  for ny,ty in zip(range(len(tempPic1)),range(len(tempPic2))):
    for nx,tx in zip(range(len(tempPic1[ny])),range(len(tempPic2[ty]))):
      tempPic1[ty][tx][0] = min(int(abs(int(tempPic1[ty][tx][0]) - int(tempPic2[ny][nx][0])*amplification)),255)
      tempPic1[ty][tx][1] = min(int(abs(int(tempPic1[ty][tx][1]) - int(tempPic2[ny][nx][1])*amplification)),255)
      tempPic1[ty][tx][2] = min(int(abs(int(tempPic1[ty][tx][2]) - int(tempPic2[ny][nx][2])*amplification)),255)
  
  return cv.cvtColor(tempPic1, cv.COLOR_RGB2BGR)

def combine(pic1:List[List[List[int]]],pic2:List[List[List[int]]], strength:float, threshold = 0) -> List[List[List[int]]]:
  tempPic1 = pic1
  tempPic2 = pic2
  for ny,ty in zip(range(len(tempPic1)),range(len(tempPic2))):
    for nx,tx in zip(range(len(tempPic1[ny])),range(len(tempPic2[ty]))):
      for ncol,tcol in zip(range(len(tempPic1[ny][nx])),range(len(tempPic2[ty][nx]))):
        temp = abs(int(float(tempPic1[ty][tx][tcol])) + int(float(tempPic2[ny][nx][ncol])))
        if(temp <= 255 and temp >= threshold):
          tempPic1[ty][tx][tcol] = temp
        elif(temp > 255):
          tempPic1[ty][tx][tcol] = 255
        else:
          tempPic1[ty][tx][tcol] = int(float(tempPic1[ty][tx][tcol]))
  return tempPic1