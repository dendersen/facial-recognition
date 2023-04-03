# import SRC.AI.simpleNeuralNetwerk.simpleAi as simpleNeuralNetwork
from SRC.image.imageCapture import Camera
from SRC.AI.siameseAI import *
from SRC.image.imageEditor import modifyOriginals

modifyOriginals()

Network = SiameseNeuralNetwork(person="David", loadAmount=500, trainDataSize=.7,bachSize=64, addFacesInTheWild=True) # Load amount should be half that of modified image lenght
prog = Network.train(10)
Network.makeAPredictionOnABatch()
Network.runSiameseModel(Camera=Camera(0), detectionThreshold=0.8, verificationThreshold=0.9)