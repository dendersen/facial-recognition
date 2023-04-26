print("loading libraries")
import tensorflow as tf
import SRC.image.imageCapture as IC
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from typing import List
from SRC.image.imageCapture import Camera
from SRC.AI.siameseAI import SiameseNeuralNetwork
import pandas as pd
from SRC.image.imageEditor import makeVarients

def chooseModel() -> int:
    answer = input()
    if (answer == "1" or answer == "2" or answer == "3"):
        print("Model " + answer + " was selected")
        return int(answer)
    else:
        print("Please select a correct model")
        chooseModel()


def takePic(camera: IC.Camera):
    print("taking pic")
    pic = camera.readCam()
    BGRface = camera.processFace(pic)
    if type(BGRface) == np.ndarray:
        # save original face
        # RGBface = cv.cvtColor(BGRface, cv.COLOR_BGR2RGB)
        BGRface = cv.resize(BGRface,(100,100))
        return BGRface
    else:
        print("ERROR: Face was not np.ndarray")

# works for simple Neural network
def identifyImage(image, model):
    # Add a batch dimension
    pic = tf.expand_dims(image, axis=0)
    
    predictions = model.predict(pic)
    score = tf.nn.softmax(predictions[0])
    
    classNames = ["Chris","David","Niels","Other"]
    winner = classNames[np.argmax(score)]
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(classNames[np.argmax(score)], 100 * np.max(score))
    )
    print(score)
    
    plt.imshow(image)
    plt.title(winner)
    plt.show()
    
    
def useKNN():
    import runKNN
    runKNN.runKNN()
    pass

def useSimpleNeuralNetwork():
    simpleNeuralNetwork = tf.keras.models.load_model('SimpleAi')
    print("model loaded")
    
    camera = IC.Camera(0)
    print("camera loaded")
    while (True):
        camera.readCam()
        if cv.waitKey(10) == 32: # wait for spacebar to be pressed
            pic = takePic(camera)
            if(pic is None):
                continue 
            
            identifyImage(pic,simpleNeuralNetwork)
            
        if cv.waitKey(10) == 27: #Stop on esc 
            print("stop")
            break

def saveData(losses: List[float], test_accuracy: List[float] = None, train_accuracy: List[float] = None, label = "Christoffer"):
    if losses == None:
        return
    
    file = open(f"Data\\{label}.csv", "a")
    for loss, test, train in zip(losses, test_accuracy, train_accuracy):
        file.write(f"{loss},{test},{train}\n")

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

def useSiameseNeuralNetwork():
    Network1 = SiameseNeuralNetwork(
    person = "Christoffer",
    loadOurData  = False,
    loadAmount = 1500,
    varients = 3,
    learning_rate = 1e-3,
    trainDataSize = .9,
    batchSize = 32,
    reprocessDataset = False,
    useDataset = True,
    resetNetwork = False,
    networkSummary = True
    )
    Network2 = SiameseNeuralNetwork(
    person = "David",
    loadOurData  = False,
    loadAmount = 1500,
    varients = 3,
    learning_rate = 1e-3,
    trainDataSize = .9,
    batchSize = 32,
    reprocessDataset = False,
    useDataset = True,
    resetNetwork = False,
    networkSummary = False
    )
    Network3 = SiameseNeuralNetwork(
    person = "Niels",
    loadOurData  = False,
    loadAmount = 1500,
    varients = 3,
    learning_rate = 1e-3,
    trainDataSize = .9,
    batchSize = 32,
    reprocessDataset = False,
    useDataset = True,
    resetNetwork = False,
    networkSummary = False
    )
    camera = Camera(0)
    while True:
        camera.readCam()
        if cv.waitKey(10) == 32: # wait for spacebar to be pressed
            image = takePic(camera)
            if(image is None):
                continue 
            image = makeVarients(image,1)[0]
            (results1, fullResult1) = Network1.runSiameseModel(Camera=None, image = image, detectionThreshold=0.5, verificationThreshold=0.5)
            if fullResult1:
                print("This is Christoffer, the results was: ", results1)
                continue
            
            (results2, fullResult2) = Network2.runSiameseModel(Camera=None, image = image, detectionThreshold=0.5, verificationThreshold=0.5)
            if fullResult2:
                print("This is David, the results was: ", results2)
                continue
            
            (results3, fullResult3) = Network3.runSiameseModel(Camera=None, image = image, detectionThreshold=0.5, verificationThreshold=0.5)
            if fullResult3:
                print("This is Niels, the results was: ", results3)
                continue
            
            print("This is not someone i know!")
        
        if cv.waitKey(10) == 27: #Stop on esc 
            print("stop")
            break

print("Welcome to our models. Please select a model: \n",
    "Press 1: KNN \n", 
    "Press 2: Simple Neural Network \n", 
    "Press 3: Siamese Neural Network")

modelNumber = chooseModel()

if modelNumber == 1:
    useKNN()

if modelNumber == 2:
    useSimpleNeuralNetwork()

if modelNumber == 3:
    useSiameseNeuralNetwork()