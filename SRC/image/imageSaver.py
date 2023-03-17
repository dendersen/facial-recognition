from PIL import Image

def saveImage(img:Image.Image,label:str, modified:bool  ,ID:int = 0):
  """saves an image to the correct location based on it's label

  Args:
      img (Image): the image that will be saved
      label (str): the subject of the image
      ID (int, optional): the first ID to be searched for in the search for the next ID, if this is too high all ID's under it will result NULL if this is low it will search for a long time to find an open ID. Defaults to 0.
  """
  path:str = ""
  
  if(modified):
    path = "images/modified/" + label  + "/"
  else:
    path = "images/original/" + label  + "/"
  
  ID = findOpenID(ID,path)
  img.save(path + str(ID))


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