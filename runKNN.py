import cv2 as cv
import numpy as np
from SRC.AI.knn.knn import Knn
from SRC.AI.knn.point import Point
from SRC.image.imageCapture import Cam
from SRC.image.imageEditor import makeVarients
import SRC.image.imageLoader as IL
import SRC.image.imageSaver as IS

def makePoint(thing: tuple[list[list[list[int]]],str]) -> Point:
  return Point([color for x in thing[0] for y in x for color in y],thing[1])

def makePoints(things: list[tuple[list[list[list[int]]],str]]):
  points:list[Point] = []
  for thing in things:
    points.append(makePoint(thing))
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
    if(["Christoffer","David","Niels","Other"].__contains__(label)):
      return label
    print("not a valid label")

def getPic():
  Camera = Cam(0)
  print("smile!")
  while True:
    pic = Camera.readCam()
    if cv.waitKey(10) == 32:
      BGRface = Camera.processFace(pic)
      if type(BGRface) == np.ndarray:
          # save original face
          RGBface = cv.cvtColor(BGRface, cv.COLOR_BGR2RGB)
          print("This is the shape of the face picture: ", RGBface.shape)
          Camera.close()
          return RGBface

IL.modifyOriginals()
perLabel:int = takeInput("the number of images to be loaded: ")
all = IL.loadImgAsArr(perLabel,False,alowModified=True)

if(perLabel*8 != len(all)):
  if(input("do you want equal number of all labels? Y/N: ") == "Y"):
    while(len(all) != perLabel*8):
      perLabel -= 1
      all = IL.loadImgAsArr(perLabel,False,alowModified=True)
    print("final number of images per label:",perLabel)

all = makePoints(all)

k = Knn(all.copy())
k.UpdateDataset([makePoint((getPic(),"UnKnown"))],[getValidLabel("who is this a picture of? ")])
print(k.testK(range(5,int(len(all)/6),2)))