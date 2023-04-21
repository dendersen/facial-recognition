from SRC.image.imageSaver import saveImage
from SRC.image.imageLoader import ProcessOther, loadImgAsArr
from typing import List, Tuple
import sys
import time
import os
import numpy as np
import random
import cv2 as cv



from SRC.image.imageSaver import saveImage

def printProgressBar(iteration, total, start_time, prefix='Progress:', suffix='Complete', length=50, fill='█'):
  total -= 1
  elapsed_time = time.time() - start_time
  progress = iteration / float(total)
  estimated_time = elapsed_time / progress if progress > 0 else 0
  remaining_time = estimated_time - elapsed_time

  percent = ('{0:.1f}').format(100 * progress)
  filled_length = int(length * iteration // total)
  bar = fill * filled_length + '-' * (length - filled_length)
  time_str = 'Remaining: {0:.1f}s'.format(remaining_time)
  count_str = '{}/{}'.format(iteration, total)

  sys.stdout.write('\r%s |%s| %s%% %s %s %s' % (prefix, bar, percent, count_str, time_str, suffix))
  sys.stdout.flush()

def noise(img: List[List[List[int]]], deviation: int) -> List[List[List[int]]]:
  noise = np.random.random_integers(low = -deviation, high=deviation, size=img.shape)
  img = img.astype(np.int32)
  
  img = cv.add(img,noise)
  img = np.clip(img, 0, 255)
  img = img.astype(np.uint8)
  return img

def changeHSB(img: List[List[List[int]]], hue: int = 0, saturation: int = 0, brightness: int = 0) -> List[List[List[int]]]:
  # Convert the input list to a NumPy array
  imgArray = np.array(img, dtype=np.uint8)
  
  # Convert the image from RGB to HSV
  hsvImage = cv.cvtColor(imgArray, cv.COLOR_RGB2HSV)

  # Add the hue, saturation, and brightness offsets
  hsvImage[:, :, 0] = (hsvImage[:, :, 0] + hue) % 180
  hsvImage[:, :, 1] = np.clip(hsvImage[:, :, 1] + saturation, 0, 255)
  hsvImage[:, :, 2] = np.clip(hsvImage[:, :, 2] + brightness, 0, 255)

  # Convert the image back to RGB
  adjustedImg = cv.cvtColor(hsvImage, cv.COLOR_HSV2RGB)

  # Convert the adjusted image back to a list of lists
  return adjustedImg.tolist()

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
  length = len(os.listdir(path))-1
  start_time = time.time()

  i = 0
  for file_name in os.listdir(path):
    # construct full file path
    file = os.path.join(path, file_name)
    if ".jpg" in file:
      os.remove(file)
      i = i+1
      printProgressBar(i, length, start_time)

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

from typing import List
import numpy as np
import cv2 as cv

def sharpen(pic: List[List[List[int]]], strength: float = 0.2, threshold: int = -1, showSteps: bool = False, showEnd: bool = False, amplification: float = 1):
    pic = np.array(pic)
    process = smooth(pic, threshold)
    if showSteps:
        cv.imshow('smooth output:', cv.convertScaleAbs(process))

    process = difference(pic, process, amplification)
    if showSteps:
        cv.imshow('detail output:', cv.convertScaleAbs(cv.multiply(process.copy(), np.ones_like(pic) * 2)))

    pic = combine(pic, process, strength, threshold)
    if showEnd:
        cv.imshow('sharp output:', cv.convertScaleAbs(pic))

    return cv.convertScaleAbs(pic)

def smooth(pic, threshold: int = -1, strong: float = 1.0, central: float = 1.0):
  orgPic = pic.copy()
  kernel = np.array([[strong, strong, strong],
                      [strong, central, strong],
                      [strong, strong, strong]]) / (central + 8 * strong)
  tempList = cv.filter2D(orgPic, -1, kernel)
  
  if threshold > 0:
    mask = (tempList - orgPic) < (threshold / 255.0)
    tempList[mask] = orgPic[mask]

  return tempList

def difference(pic1, pic2, amplification: float = 1):
  return np.clip((pic1.astype(np.float32) - pic2.astype(np.float32) * amplification), 0, 255).astype(np.uint8)

def combine(pic1, pic2, strength: float, threshold: int = 0):
  temp = np.clip(pic1.astype(np.float32) + pic2.astype(np.float32) * strength, 0, 255).astype(np.uint8)
  
  if threshold > 0:
    mask = temp < threshold
    temp[mask] = pic1[mask]

  return temp
