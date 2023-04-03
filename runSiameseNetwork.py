# import SRC.AI.simpleNeuralNetwerk.simpleAi as simpleNeuralNetwork
from SRC.image.imageCapture import Camera
from SRC.AI.siameseAI import *
from SRC.image.imageEditor import modifyOriginals

modifyOriginals(maximum=300,varients=5)
# Load amount should be half that of modified image lenght or original_lengt * 10 / 2
Network = SiameseNeuralNetwork(person="Christoffer", loadAmount=600, trainDataSize=.7, bachSize=16, addFacesInTheWild=True, resetNetwork=False) 
Network.train(10)
Network.makeAPredictionOnABatch()
Network.runSiameseModel(Camera=Camera(0), detectionThreshold=0.5, verificationThreshold=0.5) 