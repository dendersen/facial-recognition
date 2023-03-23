from SRC.AI.knn.point import Point

class Distance:
  def __init__(self,point:Point,distance:float) -> None:
    self.point:Point = point
    self.distance:float = distance

def checkDist(dist:Distance):
  return dist.distance

def getFirst(lis:list):
  return lis[0]