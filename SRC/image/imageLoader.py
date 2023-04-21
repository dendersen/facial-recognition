from PIL import Image
from random import shuffle
from numpy import array
from typing import List, Tuple

import os
import tensorflow as tf

from SRC.image.imageCapture import Camera
from SRC.image.imageSaver import saveImage
extension:str = ".jpg"

def loadImages(maxVolume:int, linearLoad:bool,labels:List[str] = ["Christoffer","David","Niels","Other"],alowModified:bool=True,alowOriginals= False,cropOri:bool = True)-> List[Tuple[Image.Image,str]]:

  """loads any number of images based on a max volume (per label) and a list of labels in use
  
  Args:
      maxVolume (int): the maximum amount of images of any one label 
      linearLoad (bool): should the images be loaded in a linear fashion or randomly loaded and shuffled
      labels (list[str]): the labels to be included in the test, default = all
      alowModified (bool): should non original images be used.
  
  Returns:
      list[tuple(Image.Image,str)]: a list containing a tuple, this tuple is accesed as such: [0] = image, [1] = label.
  """


  outgoingImages:list[Image.Image] = []
  outgoingLabels:list[str] = []

  if (linearLoad and alowOriginals):
    for label in labels:
      path:str = "images/original/" + label  + "/"
      for ID in range(0,maxVolume):
        try:
            img = Image.open(path+str(ID)+extension)
            if(cropOri):
              outgoingImages.append(img.crop((10,10,110,110)))
            else:
              outgoingImages.append(img)
            outgoingLabels.append(label)
        except:
          break

    if (linearLoad and alowModified):
      for label in labels:
        path:str = "images/modified/" + label  + "/"
        for ID in range(maxVolume):
          try:
            img = Image.open(path+str(ID)+extension)
            outgoingImages.append(img)
            outgoingLabels.append(label)
          except:
            break

  if((not linearLoad) and alowOriginals):
    for label in labels:
      path:str = "images/original/" + label  + "/"
      ID:int = 0
      tempOutgoingImages = []
      tempOutgoingLabels = []
      try:
        while (True):
          img = Image.open(path+str(ID)+extension)
          if(cropOri):
            tempOutgoingImages.append(img.crop((10,10,110,110)))
          else:
            tempOutgoingImages.append(img)
          tempOutgoingLabels.append(label)
          ID += 1
      except:
        pass
      for i in range(0,min(maxVolume,len(tempOutgoingImages))):
        outgoingImages.append(tempOutgoingImages[i])
        outgoingLabels.append(tempOutgoingLabels[i])
      i = min(maxVolume,len(tempOutgoingImages))
      while(True):
        try:
          tempOutgoingImages[i].close()
          i += 1
        except:
          break

  if((not linearLoad) and alowModified):
    for label in labels:
      path:str = "images/modified/" + label  + "/"
      ID:int = 0
      tempOutgoingImages = []
      tempOutgoingLabels = []
      try:
        while (True):
          img = Image.open(path+str(ID)+extension)
          tempOutgoingImages.append(img)
          tempOutgoingLabels.append(label)
          ID += 1
      except:
        pass
      for i in range(0,min(maxVolume,len(tempOutgoingImages))):
        outgoingImages.append(tempOutgoingImages[i])
        outgoingLabels.append(tempOutgoingLabels[i])
      i = min(maxVolume,len(tempOutgoingImages))
      while(True):
        try:
          tempOutgoingImages[i].close()
          i += 1
        except:
          break
  
  if(not linearLoad):
    out = [*zip(outgoingImages,outgoingLabels)]
    shuffle(out)
    return out
  
  return [*zip(outgoingImages,outgoingLabels)]

def loadImgAsArr(maxVolume:int, linearLoad:bool, labels: List[str] = ("Christoffer", "David", "Niels", "Other"),alowModified:bool=False,alowOriginals:bool = True,cropOri:bool = False)-> List[Tuple[List[List[List[int]]],str]]:
  image = loadImages(maxVolume, linearLoad, labels, alowModified,alowOriginals,cropOri)
  return [(array(img[0]),img[1]) for img in image]

def preprocess(filePath,label):
  
  # Read in image from file path
  byteImg = tf.io.read_file(filePath)
  # Load in image
  img = tf.io.decode_jpeg(byteImg)
  
  # preprocessing steps:
  #                   - Resize image to be 100x100x3 pixels, just to make sure
  #                   - Scale the image to be between 0 and 1
  img = tf.image.resize(img, (100,100))
  img = img/255.0
  return (img,label)

def loadDataset(loadAmount: int, trainDataSize: float = 0.7, bachSize: int = 16):
  """
  Lodes a set amount of the dataset. 
  splits the dataset into traning data and test data
  the deta is in baches
  
  Returns: (trainData, testData)
  """
  
  # Important paths to data
  chrisFolderPath = os.path.join('images/modified','Christoffer')
  davidFolderPath = os.path.join('images/modified','David')
  nielsFolderPath = os.path.join('images/modified','Niels')
  otherFolderPath = os.path.join('images/modified','Other')
  
  # makes tenserflowlist of imagepaths
  chrisImagePath = tf.data.Dataset.list_files(chrisFolderPath+'\*.jpg').take(-1)
  davidImagePath = tf.data.Dataset.list_files(davidFolderPath+'\*.jpg').take(-1)
  nielsImagePath = tf.data.Dataset.list_files(nielsFolderPath+'\*.jpg').take(-1)
  otherImagePath = tf.data.Dataset.list_files(otherFolderPath+'\*.jpg').take(-1)
  
  chrisImagePath.shuffle(10)
  davidImagePath.shuffle(10)
  nielsImagePath.shuffle(10)
  otherImagePath.shuffle(10)
  
  chrisImagePath = chrisImagePath.take(loadAmount)
  davidImagePath = davidImagePath.take(loadAmount)
  nielsImagePath = nielsImagePath.take(loadAmount)
  otherImagePath = otherImagePath.take(loadAmount)
  
  # Adds label to imagepaths
  chrisData = tf.data.Dataset.zip((chrisImagePath, tf.data.Dataset.from_tensor_slices(tf.fill(len(chrisImagePath),0)))) 
  davidData = tf.data.Dataset.zip((davidImagePath, tf.data.Dataset.from_tensor_slices(tf.fill(len(davidImagePath),1)))) 
  nielsData = tf.data.Dataset.zip((nielsImagePath, tf.data.Dataset.from_tensor_slices(tf.fill(len(nielsImagePath),2))))
  otherData = tf.data.Dataset.zip((otherImagePath, tf.data.Dataset.from_tensor_slices(tf.fill(len(otherImagePath),3))))
  
  # concatenates the data together
  chrisAndDavid = chrisData.concatenate(davidData)
  chrisDavidAndNiels = chrisAndDavid.concatenate(nielsData)
  data = chrisDavidAndNiels.concatenate(otherData)
  
  # Lodes the images
  data = data.map(preprocess)
  data = data.cache()
  data = data.shuffle(buffer_size=1024)
  
  # Training partition
  trainData = data.take(round(len(data)*trainDataSize))
  trainData = trainData.batch(bachSize)
  trainData = trainData.prefetch(8)
  
  # Testing partition
  testData = data.skip(round(len(data)*trainDataSize))
  # testData = testData.take(round(len(data)*(1-trainDataSize)))
  testData = testData.batch(bachSize)
  testData = testData.prefetch(8)
  return (trainData, testData)

def ProcessOther():
  orgPath = "images\\modified\\forDataset"
  cam = Camera(0)
  # Get all data
  img = []
  fail = 0
  progbar = tf.keras.utils.Progbar(len(os.listdir(orgPath))-1)
  for i,picture in enumerate(os.listdir(orgPath)):
    if ".jpg" in picture:
      path = os.path.join(orgPath, picture)
      temp = cam.processFace(array(Image.open(path)),False)
      if(type(temp) != type(None)):
        temp = Image.fromarray(temp)
        temp.resize((120,120))
        temp.crop((10,10,110,110))
        img.append(temp)
      else:
        fail += 1
        if(fail%10 == 0):
          print(" mistakes:" + str(fail) + " failureRate:" + str((fail/i)*100) + "%",end = "")
    progbar.update(i)
  if(fail > 0):
    print(f"there whas found {fail} pictures without a face in the dataset")
  saveImage(img,"Other",True)

