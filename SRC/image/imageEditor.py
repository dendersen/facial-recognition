from SRC.image.imageSaver import saveImage
from SRC.image.imageLoader import loadImgAsArr
from typing import List, Tuple
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

def modifyOriginals(maximum:int = 300,varients:int = 10):
  IDChris = 0
  IDDavid = 0
  IDNiels = 0

  IDOther = 0
  for image in loadImgAsArr(maximum,True,cropOri= True):
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



