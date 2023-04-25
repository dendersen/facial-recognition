import math
from SRC.image.imageSaver import saveImage
from SRC.image.imageLoader import loadImgAsArr
from typing import List, Tuple
import sys
import time
import os
import tarfile
import numpy as np
import random
import cv2 as cv
from PIL import Image
from SRC.image.imageCapture import Camera
from SRC.progBar import printProgressBar

def noise(img: List[List[List[int]]], deviation: int) -> List[List[List[int]]]:
  img = np.array(img)
  noise = np.random.random_integers(low = -deviation, high=deviation, size=img.shape)
  img = img.astype(np.int32)
  
  img = cv.add(img,noise)
  img = np.clip(img, 0, 255)
  img = img.astype(np.uint8)
  return img

def changeHSV(img: List[List[List[int]]], hue: int = 0, saturation: int = 0, brightness: int = 0) -> List[List[List[int]]]:
  # Convert the input list to a NumPy array
  imgArray = np.array(img, dtype=np.uint8)
  
  # Convert the image from RGB to HSV
  hsvImage = cv.cvtColor(imgArray, cv.COLOR_RGB2HSV)

  # Add the hue, saturation, and brightness offsets
  hsvImage[:, :, 0] += hue
  hsvImage[:, :, 0] %= 180
  hsvImage[:, :, 1] += saturation
  hsvImage[:, :, 2] += brightness

  # Convert the image back to RGB
  adjustedImg = cv.cvtColor(hsvImage, cv.COLOR_HSV2RGB)

  # Convert the adjusted image back to a list of lists
  return adjustedImg

def makeVarients(image: List[List[List[int]]], variantNumber:int = 10, move:bool = True, sharp:bool = True, hueShift:bool = True, noisy:bool = True, faces = None) -> List[List[List[List[int]]]]: 
  # Height and width is always the same. It's defined as size
  size  = image.shape[0]
  
  buffer1Size = int(size * 0.8)
  maxOffset = int(size-buffer1Size)
  if(faces == None):
    faces: list[np.ndarray[np.ndarray[np.ndarray[int]]]] = []
  
  for i in range(variantNumber):
    if(move):
      xStart =  random.randint(0,maxOffset)
      xEnd = xStart + buffer1Size
      yStart = random.randint(0,maxOffset)
      yEnd = yStart + buffer1Size
      newVariant = image[yStart:yEnd, xStart:xEnd]
    else:
      newVariant = image[10:10, 110:110]
    if (sharp):
      newVariant = sharpen(newVariant, strength = 2, size = 5 if random.random() > 0.5 else 3)
    if(hueShift):
      newVariant = changeHSV(newVariant, hue = random.randint(0,10) if random.random() < 0.9 else random.randint(5,20), saturation = 0 , brightness = 0)
    if (noisy):
      newVariant = noise(newVariant, deviation = random.randint(0,8) if random.random() < 0.995 else random.randint(5,25))
    faces.append(newVariant)
  return faces

def clearModified():
  clearPath('images\\modified\\Christoffer')
  clearPath('images\\modified\\Niels')
  clearPath('images\\modified\\David')
  clearPath('images\\modified\\Other')

def clearPath(path:str):
  print('\n Removeing images from: '+ path)
  length = len(os.listdir(path))
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
  print("clearing existing images")
  clearPath('images\\modified\\Christoffer')
  clearPath('images\\modified\\Niels')
  clearPath('images\\modified\\David')
  
  print("done clearing")
  
  IDChris = 0
  IDDavid = 0
  IDNiels = 0
  IDOther = 0
  print("loading images")
  images = loadImgAsArr(maximum,False,cropOri = False)
  print("Makeing new pictures")
  start_time = time.time()
  currentLabel = ""
  toBeSaved = []
  for i,image in enumerate(images):
    printProgressBar(i, len(images), start_time)
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
    
    if(currentLabel != image[1]):
      if(len(toBeSaved) != 0):
        saveImage(toBeSaved,currentLabel,True,ID,forceID=True,forceNoPrint = True)
      toBeSaved = []
      currentLabel  = image[1]
    toBeSaved = makeVarients(image[0],varients,faces=toBeSaved)
  
  if(dataset):
    ProcessOther()
    
  print("done")

def getLabeledFaces():
  forDatasetPath = "images\\modified\\forDataset"
  clearPath(forDatasetPath)
  # Uncompress Tar GZ Labelled faces in the wild
  with tarfile.open('lfw.tgz', "r:gz") as tar:
    print(f'\n Adding extra images to: {forDatasetPath} : From Untar Labelled Faces in the Wild Dataset')
    start_time = time.time()
    i = 0
    # Move LFW Images to the following repository data/negative
    for member in tar.getmembers():
      i = i+1
      printProgressBar(i,len(tar.getmembers()),start_time)
      if member.name.endswith(".jpg") or member.name.endswith(".png"):
        member.name = os.path.basename(member.name)
        tar.extract(member, forDatasetPath)
  ProcessOther()

def sharpen(pic: List[List[List[int]]], strength: float = 2, showSteps: bool = False, showEnd: bool = False,size:int = 3):
  pic = np.array(pic)
  coarse = gaussianKernel(pic.copy(), (size,size),2)
  if(showSteps):
    cv.imshow("coarse",coarse)
  fine = cv.subtract(pic,coarse)
  if(showSteps):
    cv.imshow("fine",np.clip(cv.multiply(fine,(np.ones_like(fine)*strength).astype(np.uint8)),0,255))
  sharp = cv.addWeighted(pic,1,fine,strength,0)
  if(showEnd):
    cv.imshow("sharp",sharp)
  return sharp

def gaussianKernel(img:List[List[List[np.uint8]]],size:Tuple[int], spread:float):
  kernel = np.zeros(size,dtype=np.float32)
  k = (size[0] - 1) // 2
  for i in range(size[0]):
    for j in range(size[1]):
      x = i - k
      y = j - k
      kernel[i, j] = (1/(2*math.pi*spread**2)) * math.exp(-(x**2 + y**2) / (2 * spread**2))
  return cv.filter2D(img, -1, kernel /kernel.sum()).astype(np.uint8)

def ProcessOther():
  clearPath('images\\modified\\Other')
  orgPath = "images\\modified\\forDataset"
  selfPath = "images\\original\\Other"
  cam = Camera(0)
  # Get all data
  img = []
  fail = 0
  paths = os.listdir(orgPath)
  paths = paths + os.listdir(selfPath)
  time_start = time.time()
  for i,picture in enumerate(paths):
    percent = ('{0:.2f}').format(100 * fail/(i+1))
    printProgressBar(i, len(paths), time_start,suffix="failure rate: " + percent + "%   ")
    if ".jpg" in picture:
      path = os.path.join(orgPath, picture)
      temp = cam.processFace(np.array(Image.open(path)),False)
      if(type(temp) != type(None)):
        temp = Image.fromarray(temp)
        temp.resize((120,120))
        temp.crop((10,10,110,110))
        img.append(makeVarients(np.array(temp),1)[0])
      else:
        fail += 1
  if(fail > 0):
    print(f"\nThere whas found {fail} pictures without a face in the dataset")
  print("Saving images\r")
  saveImage(img,"Other",True)