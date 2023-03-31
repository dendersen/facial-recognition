from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf
import os
import glob
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


# Avoid out of out of memmory errors by setting GPU Memory Consumption Growth
# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
  tf.config.experimental.set_memory_growth(gpu, True)

def rewriteDataToMatchNetwork(person: str):
  # Important paths to data
  posPath = os.path.join('images\DataSiameseNetwork','positive')
  negPath = os.path.join('images\DataSiameseNetwork','negative')
  ancPath = os.path.join('images\DataSiameseNetwork','anchor')
  
  for file_name in os.listdir(posPath):
    # construct full file path
    file = os.path.join(posPath, file_name)
    if ".jpg" in file:
      # print('Deleting file:', file)
      os.remove(file)
  
  for file_name in os.listdir(negPath):
    # construct full file path
    file = os.path.join(negPath, file_name)
    if ".jpg" in file:
      # print('Deleting file:', file)
      os.remove(file)
  
  for file_name in os.listdir(ancPath):
    # construct full file path
    file = os.path.join(ancPath, file_name)
    if ".jpg" in file:
      # print('Deleting file:', file)
      os.remove(file)
  
  # Get all negative data
  for picture in os.listdir('images\modified\Other'):
    if ".jpg" in picture:
      path = os.path.join('images\modified\Other', picture)
      img = cv.imread(path)
      newPath = os.path.join(negPath, picture)
      cv.imwrite(newPath, img)
  
  # Get all anchor data and positive data
  datapath = os.path.join('images/modified/', person)
  i = 0
  for picture in os.listdir(datapath):
    if ".jpg" in picture:
      if i % 2 == 0:
        path = os.path.join(datapath, picture)
        img = cv.imread(path)
        newPath = os.path.join(ancPath, picture)
        cv.imwrite(newPath, img)
        i = i+1
      else:
        path = os.path.join(datapath, picture)
        img = cv.imread(path)
        newPath = os.path.join(posPath, picture)
        cv.imwrite(newPath, img)
        i = i+1

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
  anchor = tf.data.Dataset.list_files(ancPath+'\*.jpg').take(loadAmount)
  positive = tf.data.Dataset.list_files(posPath+'\*.jpg').take(loadAmount)
  negative = tf.data.Dataset.list_files(negPath+'\*.jpg').take(loadAmount)
  
  # build a dataset of our data
  positives = tf.data.Dataset.zip((anchor,positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
  negatives = tf.data.Dataset.zip((anchor,negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
  data = positives.concatenate(negatives)
  
  # Build dataloader pipline
  data = data.map(preprocessTwin)
  data = data.cache()
  data = data.shuffle(buffer_size=1024)
  
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
  inp = Input(shape=(100,100,3), name='inputImage')
  
  # First block
  c1 = Conv2D(64, (10,10), activation='relu')(inp)
  m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
  
  # Second block
  c2 = Conv2D(128, (7,7), activation='relu')(m1)
  m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
  
  # Third block 
  c3 = Conv2D(128, (4,4), activation='relu')(m2)
  m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
  
  # Final embedding block
  c4 = Conv2D(256, (4,4), activation='relu')(m3)
  f1 = Flatten()(c4)
  d1 = Dense(4096, activation='sigmoid')(f1)
  
  
  return Model(inputs=[inp], outputs=[d1], name='embedding')
class L1Dist(Layer):
  
  def __init__(self, **kwargs):
    super().__init__()
  
  # compare embeddings - similarity calculation
  def call(self, inputEmbedding, validationEmbedding):
    return tf.math.abs(inputEmbedding - validationEmbedding)

def makeSiameseModel():
  
  embedding = makeImbedding()
  
  # Handle inputs
  inputImage = Input(name='inputImage', shape=(100, 100, 3)) # Anchor image input in the network
  validationImage = Input(name='ValImage', shape=(100, 100, 3)) # Validation image in the network
  
  # Combine siamese distance components
  siameseLayer = L1Dist()
  siameseLayer._name = 'distance'
  distances = siameseLayer(embedding(inputImage), embedding(validationImage))
  
  # Classification layer
  classifier = Dense(1, activation='sigmoid')(distances)
  
  return Model(inputs=[inputImage, validationImage], outputs=classifier, name='SiameseNetwork')

@tf.function # Compiles into a tensorflow graph
def trainStep(siameseNetwork,batch: int, optimizer, binaryCrossLoss):
  
  # Record all of our operations
  with tf.GradientTape() as tape: # Allows for capture of gradients from network
    
    # Get anchor and the positive/negative image
    X = batch[:2]
    # Get label
    y = batch[2]
    
    # Forward pass
    yhat = siameseNetwork(X, training=True)
    # Calculate loss
    loss = binaryCrossLoss(y, yhat)
  print("Model loss is at: ", loss)
  
  # Calculate gradients
  grad = tape.gradient(loss, siameseNetwork.trainable_variables)
  
  # Calculate updated weights and apply to siamese model
  optimizer.apply_gradients(zip(grad, siameseNetwork.trainable_variables))
  
  return loss

def train(siameseNetwork, data, EPOCHS, optimizer = tf.keras.optimizers.Adam(1e-4), binaryCrossLoss = tf.losses.BinaryCrossentropy(from_logits=True)):
  checkpointDir = './training_checkpoints'
  checkpointPrefix = os.path.join(checkpointDir, 'ckpt')
  checkpoint = tf.train.Checkpoint(opt=optimizer, siameseNetwork=siameseNetwork)
  
  # loop through epochs
  for epoch in range(1,EPOCHS+1):
    print('\n Epoch {}/{}'.format(epoch,EPOCHS))
    progbar = tf.keras.utils.Progbar(len(data))
    
    # Creating a metric object 
    r = Recall()
    p = Precision()
    
    # Loop through each batch
    for idx, batch in enumerate(data):
      # Run train step here
      loss = trainStep(siameseNetwork, batch, optimizer, binaryCrossLoss)
      yhat = siameseNetwork.predict(batch[:2])
      r.update_state(batch[2], yhat)
      p.update_state(batch[2], yhat) 
      progbar.update(idx+1)
    print("Loss is at: ",loss.numpy(), ": Recall result is at: ", r.result().numpy(), ": Precision is at: ", p.result().numpy())
    
    # Save checkpoints
    if epoch % 10 == 0: 
      checkpoint.save(file_prefix=checkpointPrefix)

def makePrediction(siameseNetwork, testInput: list, testVal: list, yTrue: list):
  yHat = siameseNetwork.predict([testInput, testVal])
  # Post processing the results 
  predictions = [1 if prediction > 0.5 else 0 for prediction in yHat]
  print("The prediction was:      ",predictions, "\n It should have guessed: ", yTrue)
  
  # Creating a metric object 
  m = Precision()
  
  # Calculating the recall value 
  m.update_state(yTrue, yHat)
  
  # Return Recall Result
  print("The model was: ",m.result().numpy(), " sure")
  
  return predictions

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
  for image in os.listdir(os.path.join('images\original', person)): # Could also be "images\modified"
    if ".jpg" in image:
      inputImg = preprocess(os.path.join('images\DataSiameseNetwork', 'inputImages', 'inputImage.jpg'))
      validationImg = preprocess(os.path.join('images\original', person, image)) # Could also be "images\modified"
      
      # Make Predictions 
      result = siameseNetwork.predict(list(np.expand_dims([inputImg, validationImg], axis=1)))
      results.append(result)
  
  # Detection Threshold: Metric above which a prediciton is considered positive 
  detection = np.sum(np.array(results) > detectionThreshold)
  
  # Verification Threshold: Proportion of positive predictions / total positive samples 
  verification = detection / len(os.listdir(os.path.join('images\original', person))) # Could also be "images\modified"
  verified: bool = verification > verificationThreshold
  
  return results, verified