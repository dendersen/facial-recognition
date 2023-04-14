# import SRC.AI.simpleNeuralNetwerk.simpleAi as simpleNeuralNetwork
from SRC.image.imageCapture import Camera
from SRC.AI.siameseAI import *

def saveData(losses:List[float], testAccuracy:List[float], trainAccuracy:List[float], label):
  file = open(f"Data\\{label}.csv","a")
  for loss,test,train in zip(losses,testAccuracy,trainAccuracy):
    file.write(f"{loss},{test},{train}\n")
  file.close()

label:str = "Christoffer"

Network = SiameseNeuralNetwork(
  person = label,
  loadAmount = 2000,
  varients = 2,
  learning_rate = 1e-6,
  trainDataSize = .8,
  bachSize = 16,
  reprocessDataset = False,
  useDataset = True,
  resetNetwork = False
) 
saveData(*Network.train(100),label)
Network.makeAPredictionOnABatch()
Network.runSiameseModel(Camera=Camera(0), detectionThreshold=0.5, verificationThreshold=0.5) 

