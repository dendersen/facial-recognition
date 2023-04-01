# import SRC.AI.simpleNeuralNetwerk.simpleAi as simpleNeuralNetwork
from SRC.image.imageCapture import Camera
from SRC.AI.siameseAI import *

NetworkChris = SiameseNeuralNetwork(person="Christoffer", loadAmount=420, trainDataSize=.7,bachSize=16)
NetworkChris.train(20)
NetworkChris.makeAPredictionOnABatch()
NetworkChris.runSiameseModel(Camera=Camera(0))