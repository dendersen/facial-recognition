from SRC.image.imageSaver import saveImage
from SRC.image.imageLoader import ProcessOther, loadImgAsArr
from typing import List, Tuple
import tensorflow as tf
import os
import numpy as np
import random
import cv2 as cv

from SRC.image.imageSaver import saveImage

def noise(img: List[List[List[int]]], deviation: int) -> List[List[List[int]]]:
  noise = np.random.random_integers(low = -deviation, high=deviation, size=img.shape)
  img = img.astype(np.int32)
  
  img = cv.add(img,noise)
  img = np.clip(img, 0, 255)
  img = img.astype(np.uint8)
  return img

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