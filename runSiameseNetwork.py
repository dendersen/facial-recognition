# import SRC.AI.simpleNeuralNetwerk.simpleAi as simpleNeuralNetwork
from SRC.image.imageCapture import Camera
from SRC.AI.siameseAI import *

# Load amount should be half that of modified image lenght or original_lengt * 10 / 2
Network = SiameseNeuralNetwork(person="Christoffer", loadAmount=250, varients=4, trainDataSize=.7, bachSize=16, reprocessDataset=False, useDataset = True, resetNetwork=False) 
Network.train(10)
Network.makeAPredictionOnABatch()
Network.runSiameseModel(Camera=Camera(0), detectionThreshold=0.5, verificationThreshold=0.5) 