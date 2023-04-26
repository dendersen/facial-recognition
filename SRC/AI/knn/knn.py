import threading
from typing import Union
from itertools import count
from SRC.AI.knn.point import Point
from SRC.AI.knn.distanceStorage import Distance as dist
from SRC.AI.knn.distanceStorage import checkDist, getFirst
from typing import List

from SRC.progBar import progBar

distances:List[dist] = []
calcBar:progBar
def clearDist():
  global distances 
  distances.clear()

def addDist(test:Point,points:List[Point],CalcID:int):
  global distances, calcBar
  for i in points:
      if i != test:
        calcBar.incriment(suffix="      ")
        distances.append(dist(i,test.distance(CalcID,i)))#calculate the distance

class Knn:
  def __init__(self,knownDataType:List[Point],k:int = 5,distID:int = 0, threads:int = 1) -> None:
    self.k = k
    self.ori:List[Point] = knownDataType#saves original know data, this ensures that you can run a test on multiple k
    self.referencePoints:List[Point] = knownDataType#contains all calculated points of differentTypes
    self.distanceCalcID = distID#which formula should be used to calculate distance
    self.threads = threads
  
  def UpdateDataset(self,data:List[Point],solution:Union[List[str], str] = "lime")->None:
    if type(solution) == str:
      solution = [solution]*(len(data))
      #in case no solution is give generates a buffer that is not meant to be read but can be read as a way to know results
    
    self.data = data #data pieces containing a List of points
    self.solution:List[str] = solution#solution to data piece of index x #true means type 1, false means type 2
    self.error = [False]*(len(self.solution))
  
  def distance(self,item0:Point,item1:Point) -> int:
    return item0.distance(self.distanceCalcID,item1)#calls the distance method contained in the class point
  
  def runData(self) -> None:#runs the algorithm through all unkown points
    progbar = progBar(len(self.data),prefix="\033[Aall calculations:")
    print("\n")
    for i,toBeTested in enumerate(self.data): #datapoint being checked
      progbar.print(i,suffix="      \n")
      
      distances:List[dist] = self.calculateDistances(toBeTested)
      
      #sorts after best
      distances.sort(key=checkDist)
      
      #finds label
      labels = [distance.point.label for l,distance in zip(range(0,self.k),distances)]#farverne af de nærmeste punkter
      existingLabels = self.findIndividualLabels(labels)
      labelCounts = [[labels.count(j),j] for j in existingLabels]#antal af hver farve
      labelCounts.sort(key=getFirst)
      
      #saves best label
      toBeTested.label = labelCounts[0][1]
      self.referencePoints.append(toBeTested)
    progbar.incriment(suffix="      \n")
    print("")
    return
  
  def findIndividualLabels(self, labels:List[str]) -> List[str]:
    foundLabels = []
    for i in labels:
      exists = False
      for j in foundLabels:
        if j == i:
          exists = True
          break
      if not exists:
        foundLabels.append(i)
    return foundLabels
  
  def calculateDistances(self, test:Point) -> List[dist]:
    global calcBar
    calcBar = progBar(len(self.referencePoints),prefix="distances calculations:")
    length = len(self.referencePoints)/self.threads
    t:List[threading.Thread] = []
    clearDist()
    for i in range(self.threads):
      t.append(threading.Thread(name="calculator:"+str(i+1)+"\\" + str(self.threads),target=addDist,args=(test,self.referencePoints[int(i*length):int((i+1)*length)],self.distanceCalcID)))
    
    for i in t:
      i.start()
    
    for i in t:
      i.join()
    
    return distances
  
  def errorRate(self,msg:str = "")->int:#counts the number of True in error array
    e=0
    for i,j in zip(self.referencePoints[::-1],self.solution[::-1]):
      if i.label != j:
        e+=1
    if(len(self.solution) == 1):
      print("\nguess =",self.referencePoints[::-1][0].label, ", correct =",self.solution[::-1][0])
      print(msg)
    else:
      print ("current K: ",self.k, "|" ,((len(self.solution) - e) / len(self.solution))*100,"percent correct\n")
    return e
  
  def testK(self,rangeOfK: range = -1) -> List[Point]:#test's for different k's on the current ori(original know points) and currently active dataset 
    if rangeOfK == -1 :#sets a default range of k
      rangeOfK = range(1,8,2)
    
    for i in rangeOfK:
      (self.buildInternalKNN(i,self.distanceCalcID))
    return
  
  def buildInternalKNN(self, k, dist):
      k_nn = Knn([*self.ori.copy()],k,dist,self.threads)#creates a new knn algorithm with a new k and dist
      k_nn.UpdateDataset(self.data.copy(),self.solution.copy())#provides the algorithem with data
      k_nn.runData()#runs the algorithm
      e = k_nn.errorRate()#checks the number of errors
      return ((e,k))#returns the errors
