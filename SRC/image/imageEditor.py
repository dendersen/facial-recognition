from SRC.image.imageSaver import saveImage
from SRC.image.imageLoader import ProcessOther, loadImgAsArr
from typing import List, Tuple
import tensorflow as tf
import os
import tarfile

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
  print("clearing existing images")
  clearPath('images\\modified\\Christoffer')
  clearPath('images\\modified\\Niels')
  clearPath('images\\modified\\David')
  
  print("done clearing")
  
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
    
  print("done")

def getLabeledFaces():
  forDatasetPath = "images\\modified\\forDataset"
  clearPath(forDatasetPath)
  # Uncompress Tar GZ Labelled faces in the wild
  with tarfile.open('lfw.tgz', "r:gz") as tar:
    print(f'\n Adding extra images to: {forDatasetPath} : From Untar Labelled Faces in the Wild Dataset')
    progbar = tf.keras.utils.Progbar(len(tar.getmembers()))
    i = 0
    # Move LFW Images to the following repository data/negative
    for member in tar.getmembers():
      i = i+1
      progbar.update(i)
      if member.name.endswith(".jpg") or member.name.endswith(".png"):
        member.name = os.path.basename(member.name)
        tar.extract(member, forDatasetPath)
  ProcessOther()

