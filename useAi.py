print("loading libraries")
import tensorflow as tf
import SRC.image.imageCapture as IC
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from typing import List
from SRC.image.imageCapture import Camera
from SRC.AI.siameseNetwork.siameseAI import SiameseNeuralNetwork
import pandas as pd
from SRC.image.imageEditor import makeVarients

# list of all the prediction the ai has made
# 1: Christoffer
# 2: David
# 3: Niels
# 4: Other
predictionsOfModels = [int]
numberOfPredictions = 0

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
        RGBface = cv.cvtColor(BGRface, cv.COLOR_BGR2RGB)
        RGBface = cv.resize(RGBface,(100,100))
        return RGBface
    else:
        print("ERROR: Face was not np.ndarray")

# works for CNN
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

    # Saves the prediction
    predictionsOfModels.append(np.argmax(score) + 1)
    
    plt.imshow(image)
    plt.title(winner)
    plt.show()
    
    
def useKNN():
    import SRC.AI.knn.runKNN as runKNN
    runKNN.runKNN()
    pass

def useCNN():
    CNN = tf.keras.models.load_model('CNNAi')
    print("model loaded")
    
    camera = IC.Camera(0)
    print("camera loaded")
    while (True):
        camera.readCam()
        if cv.waitKey(10) == 32: # wait for spacebar to be pressed
            pic = takePic(camera)
            if(pic is None):
                continue 
            pic = pic/255.0
            identifyImage(pic,CNN)
            
            
        if cv.waitKey(10) == 27: #Stop on esc 
            print(predictionsOfModels)
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
    global numberOfPredictions
    Network1 = SiameseNeuralNetwork(
    person = "Christoffer",
    loadNewData  = False,
    networkSummary = True
    )
    Network2 = SiameseNeuralNetwork(
    person = "David",
    loadNewData  = False,
    networkSummary = False
    )
    Network3 = SiameseNeuralNetwork(
    person = "Niels",
    loadNewData  = False,
    networkSummary = False
    )
    camera = Camera(0)
    while True:
        camera.readCam()
        if cv.waitKey(10) == 32: # wait for spacebar to be pressed
            image = takePic(camera)
            if(image is None):
                continue 
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            # image = makeVarients(image,1)[0]
            (results1, fullResult1) = Network1.runSiameseModel(Camera=None, image = image, detectionThreshold=0.5, verificationThreshold=0.5)
            
            (results2, fullResult2) = Network2.runSiameseModel(Camera=None, image = image, detectionThreshold=0.5, verificationThreshold=0.5)
            
            (results3, fullResult3) = Network3.runSiameseModel(Camera=None, image = image, detectionThreshold=0.5, verificationThreshold=0.5)

            sikkerhed1 = np.median(results1)
            sikkerhed2 = np.median(results2)
            sikkerhed3 = np.median(results3)
            print(f"I am {sikkerhed1*100:.1f} sure that this i Christoffer\nThe results was: {results1}\n")
            print(f"I am {sikkerhed2*100:.1f} sure that this i David\nThe results was: {results2}\n")
            print(f"I am {sikkerhed3*100:.1f} sure that this i Niels\nThe results was: {results3}\n")
            if fullResult1 or fullResult2 or fullResult3:
                if sikkerhed1 > sikkerhed2 and sikkerhed1 > sikkerhed3:
                    numberOfPredictions += 1
                    print(numberOfPredictions)
                    print("This is most likely Christoffer")
                    # Save the prediction
                    predictionsOfModels.append(1)
                elif sikkerhed2 > sikkerhed1 and sikkerhed2 > sikkerhed3:
                    numberOfPredictions += 1
                    print(numberOfPredictions)
                    print("This is most likely David")
                    predictionsOfModels.append(2)
                else:
                    numberOfPredictions += 1
                    print(numberOfPredictions)
                    print("This is most likely Niels")
                    predictionsOfModels.append(3)
            else:
                numberOfPredictions += 1
                print(numberOfPredictions)
                print("This is not someone i know!")
                predictionsOfModels.append(4)
                
            
        
        if cv.waitKey(10) == 27: #Stop on esc 
            print(predictionsOfModels)
            print("stop")
            break

print("Welcome to our models. Please select a model: \n",
    "Press 1: KNN \n", 
    "Press 2: Convolutional Neural Network \n", 
    "Press 3: Siamese Neural Network")

modelNumber = chooseModel()

if modelNumber == 1:
    useKNN()

if modelNumber == 2:
    useCNN()

if modelNumber == 3:
    useSiameseNeuralNetwork()