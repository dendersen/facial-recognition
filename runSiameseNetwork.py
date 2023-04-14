# import SRC.AI.simpleNeuralNetwerk.simpleAi as simpleNeuralNetwork
from SRC.image.imageCapture import Camera
from SRC.AI.siameseAI import *
import pandas as pd

def saveData(losses:List[float], testAccuracy:List[float], trainAccuracy:List[float], label):
  file = open(f"Data\\{label}.csv","a")
  for loss,test,train in zip(losses,testAccuracy,trainAccuracy):
    file.write(f"{loss},{test},{train}\n")
  file.close()
def modelAcc(label:str):
  #read data
  data = pd.read_csv("Data/"+label+".csv")
  
  # Show how the training went
  fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
  fig.suptitle('Training Metrics')
  
  axes[0].set_ylabel("Loss", fontsize=14)
  axes[0].plot(data["loss"])
  
  axes[1].set_ylabel("Accuracy", fontsize=14)
  axes[1].set_xlabel("Epoch", fontsize=14)
  axes[1].plot(data["trainAccuracy"], 'bo--', label = 'Train_accuracy')
  axes[1].plot(data["testAccuracy"], 'ro--', label = 'Test_accuracy')
  axes[1].legend()
  plt.show()

label:str = "Christoffer"

Network = SiameseNeuralNetwork(
  person = label,
  loadAmount = 3000,
  varients = 2,
  learning_rate = 5e-6,
  trainDataSize = .8,
  bachSize = 16,
  reprocessDataset = False,
  useDataset = True,
  resetNetwork = False
)
saveData(*Network.train(100),label)
modelAcc(label)
Network.makeAPredictionOnABatch()
Network.runSiameseModel(Camera=Camera(0), detectionThreshold=0.5, verificationThreshold=0.5) 

