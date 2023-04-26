from math import floor
from typing import List, Tuple
import cv2 as cv
import numpy as np
from SRC.AI.knn.knn import Knn
from SRC.AI.knn.point import Point
import SRC.image.imageLoader as IL
from SRC.image.imageEditor import modifyOriginals
from SRC.image.imageCapture import Camera
from SRC.progBar import progBar
from PIL import Image
def makePoint(thing: Tuple[List[List[List[int]]],str]) -> Point:
  return Point([color for x in thing[0] for y in x for color in y],thing[1])

def makePoints(things: List[Tuple[List[List[List[int]]],str]]):
  points:list[Point] = []
  print("preparing points")
  progbar = progBar(len(things))
  for i,thing in enumerate(things):
    points.append(makePoint(thing))
    progbar.print(i)
  print("there are:",len(points),"produced points")
  return points

def takeInput(msg:str)->int:
  while(True):
    try:
      return int(input(msg))
    except:
      print("NaN")

def getValidLabel(msg:str)->str:
  while(True):
    label = input(msg).capitalize()
    if(["Christoffer","David","Niels","Other","Temp"].__contains__(label)):
      return label
    print("not a valid label")

def getPic():
  Cam = Camera(0)
  print("smile!")
  while True:
    pic = Cam.readCam()
    if cv.waitKey(10) == 32:
      BGRface = Cam.processFace(pic)
      if type(BGRface) == np.ndarray:
          # save original face
          RGBface = cv.cvtColor(BGRface, cv.COLOR_BGR2RGB)
          RGBface = Image.fromarray(RGBface)
          RGBface = RGBface.crop((10,10,110,110))
          RGBface = RGBface.resize((100,100))
          return np.array(RGBface)

def getYN(msg:str) -> bool:
  return input(msg + " Y/N: ").capitalize() == "Y"

def runKNN(useOriginals:bool = None, useModified:bool = None, makeModified = False, perLabel:int = None, equal:bool = None, takePic:bool = True, distribution:float = 0.3,distID:int = 1,threadCount:int = 1):
  ori = useOriginals
  if(ori == None):
    ori = getYN("should original images be used")
  
  mod = useModified

  if(type(useModified) == type(None)):
    mod = getYN("should modified images  be used")
  elif(not ori):
    mod = True
  
  if(makeModified):
    modifyOriginals()
  
  if(perLabel == None):
    perLabel:int = takeInput("the number of images to be loaded: ")
  
  all = IL.loadImgAsArr(perLabel,False,alowModified=mod,alowOriginals=ori)
  if(equal or equal == None):
    if((perLabel*8 != len(all) and ori and mod) or (perLabel*4 != len(all) and (ori ^ mod))):
      if(equal or getYN("do you want equal number of all labels?") == "Y"):
        while((perLabel*8 != len(all) and ori and mod) or (perLabel*4 != len(all) and (ori ^ mod))):
          perLabel -= 1
          all = IL.loadImgAsArr(perLabel,False,alowModified=True)
  
  all = makePoints(all)
  print("final number of images per label:",len(all)/4)
  
  k:Knn = None
  
  if(takePic):
    k = Knn(all.copy(),distID=distID,threads=threadCount)
    k.UpdateDataset([makePoint((getPic(),"UnKnown"))],[getValidLabel("who is this a picture of? ")])
  else:
    known = all[0:floor(len(all)*(1-distribution))].copy()
    unkown = all[::-1][0:floor(len(all)*distribution)].copy()
    
    k = Knn(known,distID=distID,threads=threadCount)
    k.UpdateDataset(unkown,[i.label for i in unkown])
  print(k.testK(range(5,int(len(all)/6),2)))