from PIL import Image
from random import shuffle

def loadImages(maxVolume:int, linearLoad:bool,labels:list[str] = ["Christoffer","David","Niels"],alowModified:bool=False)-> list[tuple(Image.Image,str)]:
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

  if (linearLoad):
    for label in labels:
      path:str = "images/original/" + label  + "/"
      for ID in range(maxVolume):
        try:
          outgoingImages.append(Image.load(path + str(ID)))
          outgoingLabels.append(label)
        except:
          break

    if (linearLoad and alowModified):
      for label in labels:
        path:str = "images/modified/" + label  + "/"
        for ID in range(maxVolume):
          try:
            outgoingImages.append(Image.load(path + str(ID)))
            outgoingLabels.append(label)
          except:
            break

  if(not linearLoad):
    for label in labels:
      path:str = "images/original/" + label  + "/"
      ID:int = 0
      tempOutgoingImages = []
      tempOutgoingLabels = []
      try:
        while (True):
          tempOutgoingImages.append(Image.load(path + str(ID)))
          tempOutgoingLabels.append(label)
          ID += 1
      except:
        pass

      for i in tempOutgoingImages:
        outgoingImages.append(i)
      for i in tempOutgoingLabels:
        outgoingLabels.append(i)

  if((not linearLoad) and alowModified):
    for label in labels:
      path:str = "images/modified/" + label  + "/"
      ID:int = 0
      tempOutgoingImages = []
      tempOutgoingLabels = []
      try:
        while (True):
          tempOutgoingImages.append(Image.load(path + str(ID)))
          tempOutgoingLabels.append(label)
          ID += 1
      except:
        pass

      for i in tempOutgoingImages:
        outgoingImages.append(i)
      for i in tempOutgoingLabels:
        outgoingLabels.append(i)

  if(not linearLoad):
    return [*shuffle(zip(outgoingImages,outgoingLabels))]

  return [*zip(outgoingImages,outgoingLabels)]