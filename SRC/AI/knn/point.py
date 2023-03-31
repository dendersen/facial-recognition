from math import sqrt
from struct import pack,unpack
from typing import List
class Point:pass

class Point:
  def __init__(self,location:list[float],label:str = "lime") -> None:
    self.location = location
    self.label = label
    self.dist = [self.euclid,self.manhattan,self.chebyshev,self.hammingManhattan,self.hammingEuclid,self.hammingChebyshev]
  
  def removeLabel(self) -> Point:
    label = "lime"
    return self
  
  def distance(self,version:int,point:Point):
    return self.dist[version](point)
  def euclid(self,point:Point)->float:
    return sqrt(sum([(i-l)**2 for i,l in zip(self.location,point.location)]))
  def manhattan(self,point:Point)->float:
    return abs((sum([(i-l) for i,l in zip(self.location,point.location)])))
  def chebyshev(self,point:Point)->float:
    return max([abs(i-l) for i,l in zip(self.location,point.location)])
  def hamming(self,point:Point) -> List[int]:
    differ = []
    for i,l in (self.location,point.location):
      differ.append(dif(floatToBin(i),floatToBin(l)))
    return differ
  def hammingManhattan(self,point:Point)->float:
    return sum(self.hamming(point))
  def hammingEuclid(self,point:Point):
    return sqrt(sum(i**2 for i in self.hamming(point)))
  def hammingChebyshev(self,point:Point):
    return max(self.hamming(point))

def floatToBin(F:float)->str:
  return bin(unpack("!i",pack("!f",F))[0]).replace("0b","")

def dif(i:list,j:list):
  Dif = 0
  for l,L in zip(i,j):
    Dif += l != L
  return Dif  
