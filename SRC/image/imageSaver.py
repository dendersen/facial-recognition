from typing import Union
from PIL import Image
from typing import List

extension:str = ".jpg"
def saveImage(img:Union[List[Image.Image], List[List[List[List[int]]]],Image.Image], label:str, modified:bool ,ID:int = 0,forceID:bool = False, forceNoPrint:bool = False) -> None:
  """saves an image to the correct location based on it's label
  
  Args:
      img (Union[list[Image.Image], list[list[list[list[int]]]]): the image that will be saved
      label (str): the subject of the image
      ID (int, optional): the first ID to be searched for in the search for the next ID, if this is too high all ID's under it will result NULL if this is low it will search for a long time to find an open ID. Defaults to 0.
  """
  if(type(img) == type(None)):
    return
  if type(img) == Image.Image:
    img = [img]
  elif(len(img) > 0 and type(img[0]) != Image.Image):
    temp:list[Image.Image] = []
    for i in img:
      temp.append(arrToPIL(i))
    img = temp
  elif(len(img) == 0):
    return
  img:list[Image.Image]
  
  path:str = ""
  if(modified):
    path = "images/modified/" + label  + "/"
  else:
    path = "images/original/" + label  + "/"
  
  if not forceID:
    ID = findOpenID(ID,path)
  for image in img:
    image = imageScale(image,modified)
    image.save(path + str(ID) + ".jpg")
    ID += 1
  
  if((not forceNoPrint) and len(img) > 5):
    print(f"{len(img)} images where saved")
  
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
      img = Image.open(path + str(ID)+extension)
      img.close()
      ID += 1
    except:
      return ID

def imageScale(img:Image.Image, isModified:bool, desiredWidth:int = 100, desiredHeight:int = 100) -> Image.Image:
  """scales the input image to the desired size using linear scaling
  
  Args:
      img (Image.Image): the image to be scaled
      desiredWidth (int, optional): the new width of the image. Defaults to 100.
      desiredHeight (int, optional): the new height of the image. Defaults to 100.
  
  Returns:
      Image.Image: the scaled image
  """
  if not isModified:
    return img.resize((120,120))
  else:
    return img.resize((100,100))

def arrToPIL(img: List[List[List[int]]]):
  return Image.fromarray(img)
