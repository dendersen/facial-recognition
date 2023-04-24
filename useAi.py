print("loading libraries")
import tensorflow as tf
import SRC.image.imageCapture as IC
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

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

def useSiameseNeuralNetwork():
    pass

print("Welcome to our models. Please select a model: \n",
      "Press 1: KNN \n", 
      "Press 2: Simple Neural Network \n", 
      "Press 3: Siamese Neural Network")

modelNumber = chooseModel()

if modelNumber == 1:
    useKNN()
    pass

if modelNumber == 2:
    useSimpleNeuralNetwork()

if modelNumber == 3:
    pass



