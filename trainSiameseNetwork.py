# import SRC.AI.simpleNeuralNetwerk.simpleAi as simpleNeuralNetwork
from typing import List
from matplotlib import pyplot as plt
from SRC.AI.siameseAI import SiameseNeuralNetwork
import pandas as pd
from typing import Union

def saveData(losses: Union[List[float],None], test_accuracy: Union[List[float],None] = None, train_accuracy: Union[List[float],None] = None, label = "Christoffer"):
  if losses == None:
    return
  
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

label:str = "Niels"

Network = SiameseNeuralNetwork(
  person = label,
  loadOurData  = True,
  loadAmount = 2000,
  varients = 5,
  learning_rate = 1e-3,
  trainDataSize = .9,
  batchSize = 64,
  reprocessDataset = False,
  useDataset = True,
  resetNetwork = False,
  networkSummary = False
)

saveData(*Network.train(50),label=label)
modelAcc(label)
Network.makeAPredictionOnABatch()
