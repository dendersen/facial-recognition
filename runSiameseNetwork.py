# import SRC.AI.simpleNeuralNetwerk.simpleAi as simpleNeuralNetwork
from typing import List
from matplotlib import pyplot as plt
from SRC.image.imageCapture import Camera
from SRC.AI.siameseAI import SiameseNeuralNetwork
import pandas as pd

def saveData(losses: List[float], test_accuracy: List[float], train_accuracy: List[float], label):
  file = open(f"Data\\{label}.csv", "a")
  for loss, test, train in zip(losses, test_accuracy, train_accuracy):
    file.write(f"{loss},{test},{train}\n")
  file.close()

def modelAcc(label: str):
  data = pd.read_csv("Data/" + label + ".csv")
  
  # Show how the training went
  fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
  fig.suptitle('Training Metrics')
  
  axes[0].set_ylabel("Loss", fontsize=14)
  axes[0].plot(data["loss"])
  
  axes[1].set_ylabel("Accuracy", fontsize=14)
  axes[1].set_xlabel("Epoch", fontsize=14)
  axes[1].plot(data["trainAccuracy"], 'bo--', label='Train_accuracy')
  axes[1].plot(data["testAccuracy"], 'ro--', label='Test_accuracy')
  axes[1].legend()
  plt.show()

label:str = "David"

Network = SiameseNeuralNetwork(
  person = label,
  loadAmount = 1500,
  varients = 3,
  learning_rate = 1e-5,
  trainDataSize = .9,
  batchSize = 16,
  reprocessDataset = False,
  useDataset = True,
  resetNetwork = True
)

saveData(*Network.train(20),label)
# modelAcc(label)
Network.makeAPredictionOnABatch()
# Network.runSiameseModel(Camera=Camera(0), detectionThreshold=0.5, verificationThreshold=0.5) 
