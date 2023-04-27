from typing import List
from matplotlib import pyplot as plt
from SRC.AI.siameseAI import SiameseNeuralNetwork
import pandas as pd
from typing import Union
import itertools
from SRC.progBar import progBar
import os
import platform

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

def clear_screen():
    if platform.system().lower() == "windows":
        os.system("cls")
    else:
        os.system("clear")

def trainAndEvaluate(learningRate, lamdaReg, label, EPOCHS=50, loadNewData = False, resetNetwork = True):
  network = SiameseNeuralNetwork(
    person=label,
    loadNewData=loadNewData,
    loadAmount=2000,
    varients=5,
    learningRate=learningRate,
    lamdaReg=lamdaReg,
    trainDataSize=0.9,
    batchSize=64,
    reprocessDataset=False,
    useDataset=True,
    resetNetwork=resetNetwork,
    networkSummary=False
  )
  
  trainLossResults,testAccuracyResults,trainAccuracyResults = network.train(EPOCHS)
  saveData(trainLossResults,testAccuracyResults,trainAccuracyResults, label=label)
  return testAccuracyResults[-1]

def findBesthyperparameters():
  # Definer intervallerne for hyperparametre
  learning_rate_values = [1e-2, 1e-3, 1e-4, 1e-5]
  lambdaReg_values = [1e-3, 1e-4, 1e-5, 1e-6]

  # Kombiner alle mulige værdier af hyperparametre
  hyperparameter_combinations = list(itertools.product(learning_rate_values, lambdaReg_values))

  # Variabler til at holde den bedste kombination og resultater
  best_hyperparameters = None
  best_val_accuracy = 0

  label = "Niels"
  EPOCHS = 10
  progbar = progBar(len(learning_rate_values)*len(lambdaReg_values),prefix="\033[4Aall calculations:")
  for i, var in enumerate(hyperparameter_combinations):
    if(i == 1):
      # Call the function to clear the terminal
      clear_screen()
      progbar.print(i,suffix="\n\n")
    lr, lambdaReg = var
    # progbar.print(i,suffix="\n")
    print(f"Training with learning rate: {lr}, lambdaReg: {lambdaReg}")
    # Træn og evaluer modellen med de aktuelle hyperparametre
    val_accuracy = trainAndEvaluate(lr, lambdaReg, label, EPOCHS)

    # Hvis nøjagtigheden på valideringsdata er bedre end den hidtil bedste, opdater den bedste kombination og resultater
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_hyperparameters = (lr, lambdaReg)
    progbar.print(i+1,suffix="\n\n")
  print(f"Best hyperparameters: learning_rate: {best_hyperparameters[0]}, lambdaReg: {best_hyperparameters[1]}")
label = "Christoffer"
trainAndEvaluate(learningRate=0.0001,lamdaReg=1e-6, label=label, EPOCHS=10, loadNewData=True, resetNetwork=False)
modelAcc(label)