# import SRC.AI.simpleNeuralNetwerk.simpleAi as simpleNeuralNetwork
from SRC.image.imageCapture import Camera
from SRC.AI.siameseAI import *

def saveData(losses:list[float], testAccuracy:list[float], trainAccuracy:list[float], label):
  for loss,test,train in zip(losses,testAccuracy,trainAccuracy):
    file = open(f"Data\\{label}.csv","a")
    file.write(f"{loss},{test},{train}\n")
    file.close()

label:str = "Christoffer"

# Load amount should be half that of modified image lenght or original_lengt * 10 / 2
Network = SiameseNeuralNetwork(person=label, loadAmount=250, varients=4, trainDataSize=.7, bachSize=16, reprocessDataset=False, useDataset = True, resetNetwork=False) 
saveData(*Network.train(10),label)
Network.makeAPredictionOnABatch()
Network.runSiameseModel(Camera=Camera(0), detectionThreshold=0.5, verificationThreshold=0.5) 

