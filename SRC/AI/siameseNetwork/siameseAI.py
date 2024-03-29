import os
import random
import math
from typing import List

import cv2 as cv
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from SRC.progBar import progBar

from SRC.image.imageEditor import clearPath, makeVarients, modifyOriginals, getLabeledFaces
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.python.keras.metrics import Precision

# Avoid out of memory errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
  tf.config.experimental.set_memory_growth(gpu, True)

def rewriteDataToMatchNetwork(person: str, reprocessDataset: bool = False):
  # Important paths to data
  posPath = os.path.join('images\DataSiameseNetwork','positive')
  negPath = os.path.join('images\DataSiameseNetwork','negative')
  ancPath = os.path.join('images\DataSiameseNetwork','anchor')
  
  clearPath(posPath)
  clearPath(negPath)
  clearPath(ancPath)
  # Get exstra negative data, from Untar Labelled Faces in the Wild Dataset
  if reprocessDataset:
    getLabeledFaces()
  
  # Get all modified negative data
  for name in ["Christoffer","Niels","David","Other"]:
    if name != person:
      print('\n Adding images to: '+ negPath +' : from: images\\modified\\' + name)
      progbar = progBar(len(os.listdir('images\\modified\\' + name))-1)
      for i,picture in enumerate(os.listdir('images\\modified\\' + name)):
        if ".jpg" in picture:
          path = os.path.join('images\modified', name, picture)
          img = cv.imread(path)
          newPath = os.path.join(negPath, name + "_" + picture)
          cv.imwrite(newPath, img)
          progbar.print(i+1)
  
  # Get all original negative data
  for name in ["Christoffer","Niels","David","Other"]:
    if name != person:
      print('\n Adding images to: '+ negPath +' : from: images\\original\\' + name)
      progbar = progBar(len(os.listdir('images\\original\\' + name))-1)
      for i,picture in enumerate(os.listdir('images\\original\\' + name)):
        if ".jpg" in picture:
          path = os.path.join('images\original', name, picture)
          img = cv.imread(path)
          newPath = os.path.join(negPath, name + "_" + picture)
          cv.imwrite(newPath, img)
          progbar.print(i+1)
  
  # Add all modified anchor data and positive data
  datapath = os.path.join('images/modified/', person)
  
  print('\n Adding images to: '+ posPath +' and '+ ancPath +' : from: '+ datapath)
  progbar = progBar(len(os.listdir(datapath)))
  for i,picture in enumerate(os.listdir(datapath)):
    if ".jpg" in picture:
      path = os.path.join(datapath, picture)
      newPath = os.path.join(ancPath, picture)
      if i % 2 == 0:
        img = cv.imread(path)
      else:
        newPath = os.path.join(posPath, picture)
      cv.imwrite(newPath, img)
    progbar.print(i+1)
  
  # Add all original anchor data and positive data
  datapath = os.path.join('images/original/', person)
  
  print('\n Adding images to: '+ posPath +' and '+ ancPath +' : from: '+ datapath)
  progbar = progBar(len(os.listdir(datapath)))
  for i, picture in enumerate(os.listdir(datapath)):
    if ".jpg" in picture:
      path = os.path.join(datapath, picture)
      newPath = os.path.join(ancPath, picture)
      if i % 2 == 0:
        img = cv.imread(path)
      else:
        newPath = os.path.join(posPath, picture)
      cv.imwrite(newPath, img)
    progbar.print(i+1)

# loades the image
def preprocess(filePath):
  # Read in image from file path
  byteImg = tf.io.read_file(filePath)
  # Load in image
  img = tf.io.decode_jpeg(byteImg)
  
  # preprocessing steps:
  #                   - Resize image to be 100x100x3 pixels, just to make sure
  #                   - Scale the image to be between 0 and 1
  img = tf.image.resize(img, (100,100))
  img = img/255.0
  return img

def preprocessTwin(inputImg, validationImg, label):
  return(preprocess(inputImg), preprocess(validationImg), label)

def buildData(loadAmount: int = 300, trainDataSize: float = 0.7, bachSize: int = 16):
  # Important paths to data
  posPath = os.path.join('images\DataSiameseNetwork','positive')
  negPath = os.path.join('images\DataSiameseNetwork','negative')
  ancPath = os.path.join('images\DataSiameseNetwork','anchor')
  
  # load data
  anchor = tf.data.Dataset.list_files(ancPath+'\*.jpg', shuffle=True).take(loadAmount)
  positive = tf.data.Dataset.list_files(posPath+'\*.jpg', shuffle=True).take(loadAmount)
  negative = tf.data.Dataset.list_files(negPath+'\*.jpg', shuffle=True).take(loadAmount)
  
  # build a dataset of our data
  positives = tf.data.Dataset.zip((anchor,positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
  negatives = tf.data.Dataset.zip((anchor,negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
  data = positives.concatenate(negatives)
  
  # Build dataloader pipline
  data = data.map(preprocessTwin)
  data = data.cache()
  data = data.shuffle(buffer_size=len(data))
  
  # Training partition
  trainData = data.take(round(len(data)*trainDataSize))
  trainData = trainData.batch(bachSize)
  trainData = trainData.prefetch(8)
  
  # Testing partition
  testData = data.skip(round(len(data)*trainDataSize))
  # testData = testData.take(round(len(data)*(1-trainDataSize)))
  testData = testData.batch(bachSize)
  testData = testData.prefetch(8)
  return (trainData, testData)

def makeImbedding():
  input_image = Input(shape=(100, 100, 3), name='inputImage')
  
  # First block
  conv1 = Conv2D(64, (10, 10), activation='relu')(input_image)
  maxpool1 = MaxPooling2D(pool_size = (2, 2), padding = 'same')(conv1)
  
  # Second block
  conv2 = Conv2D(128, (7, 7), activation='relu')(maxpool1)
  maxpool2 = MaxPooling2D(pool_size = (2, 2), padding = 'same')(conv2)
  
  # Third block
  conv3 = Conv2D(128, (4, 4), activation='relu')(maxpool2)
  maxpool3 = MaxPooling2D(pool_size = (2, 2), padding = 'same')(conv3)
  
  # Final embedding block
  conv4 = Conv2D(256, (4, 4), activation='relu')(maxpool3)
  flat1 = Flatten()(conv4)
  dense1 = Dense(4096, activation='sigmoid')(flat1)
  
  return Model(inputs=[input_image], outputs=[dense1], name='embedding')

class L1Dist(Layer):
  
  def __init__(self, name=None, **kwargs):
    super().__init__(name=name, **kwargs)
  
  # compare embeddings - similarity calculation
  def call(self, inputEmbedding, validationEmbedding):
    return tf.math.abs(inputEmbedding - validationEmbedding)

# Make an new instance of a model
def makeSiameseModel():
  embedding = makeImbedding()
  
  # Handle inputs
  inputImage = Input(name='inputImage', shape=(100, 100, 3)) # Anchor image input in the network
  validationImage = Input(name='ValImage', shape=(100, 100, 3)) # Validation image in the network
  
  # Combine siamese distance components
  siameseLayer = L1Dist()
  siameseLayer._name = 'L1Dist'
  distances = siameseLayer(embedding(inputImage), embedding(validationImage))
  
  # Classification layer
  classifier = Dense(1, activation='sigmoid')(distances)
  
  return Model(inputs=[inputImage, validationImage], outputs=classifier, name='SiameseNetwork')

def verify(siameseNetwork, detectionThreshold: float = 0.5, verificationThreshold: float = 0.5, person: str = "Christoffer"):
  """Verify if a given detection is the same as positive in the model
  Args:
      siameseNetwork (network): The model in use
      detectionThreshold (float): A metric above which a prediction is considered positive
      verificationThreshold (float): Proportion of positive predictions / total positive samples
      person (str): The person you want to detect, default is Christoffer
  Returns:
      results, verified
  """
  
  # Build results array
  results = []
  for image in os.listdir('images\\DataSiameseNetwork\\verificationImages'):
    if ".jpg" in image:
      inputImg = preprocess(os.path.join('images\DataSiameseNetwork', 'inputImages', 'inputImage.jpg'))
      validationImg = preprocess(os.path.join('images\\DataSiameseNetwork\\verificationImages', image))
      
      # Make Predictions 
      result = siameseNetwork.predict(list(np.expand_dims([inputImg, validationImg], axis=1)))
      results.append(result[0][0])
  
  # Detection Threshold: Metric above which a prediciton is considered positive 
  detection = np.sum(np.array(results) > detectionThreshold)
  
  # Verification Threshold: Proportion of positive predictions / total positive samples 
  verification = detection / len(os.listdir('images\\DataSiameseNetwork\\verificationImages')) # Could also be "images\modified"
  verified: bool = verification > verificationThreshold
  
  return results, verified

def showSiameseBatch(testInput, testVal, yTrue, yHat, person):
  inputImages = testInput
  testImages = testVal

  height = math.ceil((len(inputImages) + len(testImages)) / 4)
  fig, axs = plt.subplots(4, height, figsize=(16, 18))
  indexInput = 0
  indexTest = 0
  fig.suptitle(f"The person we are looking at is: {person} : The first row shows what should be guessed, the second shows the guess", fontsize=14, fontweight='bold')
  for i in range(4):
    for j in range(height):
      if i % 2 == 0:
        axs[i, j].imshow(inputImages[indexInput])
        axs[i, j].set_title(str(yTrue[indexInput]))
        axs[i, j].axis("off")
        indexInput = indexInput + 1
      else:
        axs[i, j].imshow(testImages[indexTest])
        axs[i, j].set_title(str(round(yHat[indexTest][0], 4)))
        axs[i, j].axis("off")
        indexTest = indexTest + 1
  plt.show()

def binaryCrossentropyWithRegularization(yTrue, yPred, weights, lambdaReg=1e-4):
  crossEntropyLoss = tf.keras.losses.binary_crossentropy(yTrue, yPred)
  l2Reg = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * lambdaReg
  return crossEntropyLoss + l2Reg, crossEntropyLoss

class SiameseNeuralNetwork:
  def __init__(self, person: str = "Christoffer", loadNewData:bool = True, loadAmount: int = 2000, varients:int = 5, learningRate: float = 0.0001, lamdaReg: float = 1e-6, trainDataSize: float = 0.9, batchSize: int = 64, reprocessDataset: bool = False, useDataset:bool = True, resetNetwork: bool = False, networkSummary:bool = True):
    self.person: str = person
    
    if(not useDataset):
      clearPath("images\modified\Other")
    
    self.loadNewData = loadNewData
    if loadNewData:
      modifyOriginals(6000,varients)
      rewriteDataToMatchNetwork(person=self.person, reprocessDataset = reprocessDataset)
    # Get data from files
    (self.trainingData, self.testData) = buildData(loadAmount=loadAmount,trainDataSize=trainDataSize,bachSize=batchSize)
    self.trainingData = self.trainingData.prefetch(tf.data.experimental.AUTOTUNE)
    self.testData = self.testData.prefetch(tf.data.experimental.AUTOTUNE)
    
    # Optimizer and loss
    self.lossObject = binaryCrossentropyWithRegularization
    self.lamdaReg = lamdaReg
    self.optimizer = tf.keras.optimizers.Adam(learningRate)
    
    if resetNetwork:
      self.siameseNetwork = makeSiameseModel()
    else:
      # Reload model 
      self.siameseNetwork = tf.keras.models.load_model("siamesemodelBest" + self.person)
    
    if networkSummary:
      # Gives a summary of the network
      self.siameseNetwork.summary()
  
  def train(self, EPOCHS: int = 10) -> List[List[float]]:
    if (not self.loadNewData):
      print("Remember to check if you have loaded the data you want!")
    
    
    # Keep results for plotting
    trainLossResults = []
    trainAccuracyResults = []
    testAccuracyResults = []
    
    # loss funktion
    @tf.function
    def loss(images, labels, training):
      # training=training is needed only if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      predictions = self.siameseNetwork(images, training=training)
      labels = tf.expand_dims(labels, axis=-1)
      return self.lossObject(yTrue=labels, yPred=predictions, weights=self.siameseNetwork.trainable_variables, lambdaReg=self.lamdaReg)
    
    # gets the gradiants
    @tf.function # Compiles into a tensorflow graph
    def grad(batch: int):
      # Record all of our operations
      with tf.GradientTape() as tape: # Allows for capture of gradients from network
        
        # Get anchor and the positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]
        
        lossValue, crossEntropyLoss = loss(images=X,labels=y,training=True)
        
      return tape.gradient(lossValue, self.siameseNetwork.trainable_variables), crossEntropyLoss
    
    # loop through epochs
    for epoch in range(EPOCHS):
      
      epochLossAvg = tf.keras.metrics.Mean()
      epochAccuracy = tf.keras.metrics.BinaryAccuracy()
      testepochAccuracy = tf.keras.metrics.BinaryAccuracy()
      
      print('\033[3AEpoch {}/{}                                          ' .format(epoch+1,EPOCHS))
      progbar = progBar(len(self.trainingData))
      progbar.print(0)
      # Loop through each batch
      for idx, batch in enumerate(self.trainingData):
        # Run train step here
        grads, crossEntropyLoss = grad(batch)
        self.optimizer.apply_gradients(zip(grads, self.siameseNetwork.trainable_variables))
        
        # Track progress
        epochLossAvg.update_state(crossEntropyLoss)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        X = batch[:2]
        y = batch[2]
        epochAccuracy.update_state(y, self.siameseNetwork(X, training=True))
        # Update progbar
        currentLoss = epochLossAvg.result().numpy()
        currentAccuracy = epochAccuracy.result().numpy()
        progbar.print(idx+1, suffix=f"crossEntropyLoss: {currentLoss:.3f}, Accuracy: {currentAccuracy:.3%}   ")
      
      for batch in self.testData:
        X = batch[:2]
        y = batch[2]
        # Test the mode on the test data
        # training=False is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        testepochAccuracy.update_state(y, self.siameseNetwork(X, training=False))
      
      # End epoch
      trainLossResults.append(epochLossAvg.result())
      trainAccuracyResults.append(epochAccuracy.result())
      testAccuracyResults.append(testepochAccuracy.result())
      
      if epoch % 1 == 0:
        print("\nEpoch {:03d}: crossEntropyLoss: {:.3f}, Accuracy: {:.3%}, Test accuracy: {:.3%}".format(epoch,
                                                                      epochLossAvg.result(),
                                                                      epochAccuracy.result(),
                                                                      testepochAccuracy.result()))
    
    # Show how the training went
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')
    
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(trainLossResults)
    
    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(trainAccuracyResults, 'bo--', label = 'Train_accuracy')
    axes[1].plot(testAccuracyResults, 'ro--', label = 'Test_accuracy')
    axes[1].legend()
    plt.show()
    self.siameseNetwork.save("siamesemodelBest" + self.person, save_format='tf')
    return trainLossResults,testAccuracyResults,trainAccuracyResults
  
  # Makes some predictions on some data and outputs how sure it was
  def makeAPredictionOnABatch(self):
    if (not self.loadNewData):
      print("Remember to check if you have loaded the data you want!")
    testInput, testVal, yTrue = self.testData.as_numpy_iterator().next()
    yHat = self.siameseNetwork.predict([testInput, testVal])
    # Post processing the results 
    predictions = [1 if prediction > 0.5 else 0 for prediction in yHat]
    print("The prediction was:      ",predictions, "\n It should have guessed: ", yTrue)
    
    # Creating a metric object 
    m = Precision()
    
    # Calculating the recall value 
    m.update_state(yTrue, yHat)
    
    # Return Recall Result
    print("The model was: ",m.result().numpy(), " sure")
    
    showSiameseBatch(testInput, testVal, yTrue, yHat, self.person)
    
    return predictions
  
  def runSiameseModel(self, Camera, image, detectionThreshold: float = 0.5, verificationThreshold: float = 0.5):
    """Runs the siamese model you are currently working on
    Args:
        Camera: Takes an instance of the class Cam from imageCapture, e.g Cam(0)
        Image: Takes in a picture of a person to verify
    Returns:
      True if you are verified and false if not
    """
    verificationPath = 'images\\DataSiameseNetwork\\verificationImages'
    # Clear varificationImages
    clearPath(verificationPath)
    
    # Get verify images
    pathToImagesFromPerson = os.path.join('images\original', self.person)
    print('\n Adding 10 random images to: '+ verificationPath +' : from: '+ pathToImagesFromPerson)
    
    highestID = len(os.listdir(pathToImagesFromPerson))-2
    
    for i in range(10):
      path = os.path.join(pathToImagesFromPerson, str(random.randint(0,highestID))+'.jpg')
      img = cv.imread(path)
      newPath = os.path.join(verificationPath, str(len(os.listdir(verificationPath)))+'.jpg')
      cv.imwrite(newPath, img)
    
    if (not Camera == None):
      # Initialize a Cam class
      Camera = Camera
      
      while True:
        frame = Camera.readCam()
        
        # Verification trigger
        if cv.waitKey(10) & 0xFF == ord('v'):
          face = Camera.processFace(frame)
          if type(face) != type(None):
            face = makeVarients(face,1)[0]
            if type(face) == np.ndarray:
              cv.imwrite(os.path.join('images\DataSiameseNetwork', 'inputImages', 'inputImage.jpg'), face)
              # Run verification
              results, verified = verify(self.siameseNetwork, detectionThreshold, verificationThreshold, person=self.person)
              if verified:
                print("This is " + self.person)
                print("The results are: ", results)
              else:
                print("This is not " + self.person)
                print("The results are: ", results)
        
        if cv.waitKey(10) & 0xFF == ord('q'):
          break
      cv.destroyAllWindows()
    
    if type(image) == np.ndarray: 
      cv.imwrite(os.path.join('images\DataSiameseNetwork', 'inputImages', 'inputImage.jpg'), image)
      # Run verification
      results, verified = verify(self.siameseNetwork, detectionThreshold, verificationThreshold, person=self.person)
      
      return (results, verified)