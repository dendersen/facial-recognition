from typing import Union
from PIL import Image

def saveImage(img:Union[Image.Image, list[list[list[int]]]], label:str, modified:bool  ,ID:int = 0) -> None:
  """saves an image to the correct location based on it's label

  Args:
      img (Image.Image): the image that will be saved
      label (str): the subject of the image
      ID (int, optional): the first ID to be searched for in the search for the next ID, if this is too high all ID's under it will result NULL if this is low it will search for a long time to find an open ID. Defaults to 0.
  """
  
  if(type(img) != Image.Image):
    img:Image.Image = arrToPIL(img)
  img:Image.Image
  
  path:str = ""
  if(modified):
    path = "images/modified/" + label  + "/"
  else:
    path = "images/original/" + label  + "/"
  
  ID = findOpenID(ID,path)
  img = imageScale(img)
  img.save(path + str(ID),"jpg")
  img.show()
  
  return

def findOpenID(ID:int,path:str) -> int:
  """finds the next unused ID from the given ID

  Args:
      ID (int): the first ID to be checked
      path (str): the location to search for the images

  Returns:
      int: an unused ID
  """
  while (True):
    try:
      Image.open(path + str(ID))
      ID += 1
    except:
      return ID

def imageScale(img:Image.Image, desiredWidth:int = 100, desiredHeight:int = 100) -> Image.Image:
  """scales the input image to the desired size using linear scaling

  Args:
      img (Image.Image): the image to be scaled
      desiredWidth (int, optional): the new width of the image. Defaults to 100.
      desiredHeight (int, optional): the new height of the image. Defaults to 100.

  Returns:
      Image.Image: the scaled image
  """
  img = img.resize((desiredWidth*(6/5),desiredHeight*(6/5)))
  img = img.crop(10,10,desiredWidth+10,desiredHeight+10)
  return img

def arrToPIL(img: list[list[list[int]]]):
  return Image.fromarray(img)
