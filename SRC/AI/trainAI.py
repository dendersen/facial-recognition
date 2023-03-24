from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf
import os
import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np

# Avoid out of out of memmory errors by setting GPU Memory Consumption Growth
# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
  tf.config.experimental.set_memory_growth(gpu, True)

# Important paths to data
posPath = os.path.join('images\DataSiameseNetwork','positive')
negPath = os.path.join('images\DataSiameseNetwork','negative')
ancPath = os.path.join('images\DataSiameseNetwork','anchor')

def rewriteDataToMatchNetwork(person):
  for file_name in os.listdir(posPath):
    # construct full file path
    file = os.path.join(posPath, file_name)
    if ".jpg" in file:
      print('Deleting file:', file)
      os.remove(file)
  
  for file_name in os.listdir(negPath):
    # construct full file path
    file = os.path.join(negPath, file_name)
    if ".jpg" in file:
      print('Deleting file:', file)
      os.remove(file)
  
  for file_name in os.listdir(ancPath):
    # construct full file path
    file = os.path.join(ancPath, file_name)
    if ".jpg" in file:
      print('Deleting file:', file)
      os.remove(file)
  
  
  # Get all negative data
  for picture in os.listdir('images\modified\Other'):
    if ".jpg" in picture:
      path = os.path.join('images\modified\Other', picture)
      img = cv2.imread(path)
      newPath = os.path.join(negPath, picture)
      cv2.imwrite(newPath, img)
  
  # Get all anchor data and positive data
  datapath = os.path.join('images/modified/', person)
  datapath2 = os.path.join('images/modified/', person)
  i = 0
  for picture in os.listdir(datapath):
    if ".jpg" in picture:
      if i % 2 == 0:
        path = os.path.join(datapath, picture)
        img = cv2.imread(path)
        newPath = os.path.join(ancPath, picture)
        cv2.imwrite(newPath, img)
        i = i+1
      else:
        path = os.path.join(datapath2, picture)
        img = cv2.imread(path)
        newPath = os.path.join(posPath, picture)
        cv2.imwrite(newPath, img)
        i = i+1

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

def buildData():
  # load data
  anchor = tf.data.Dataset.list_files(ancPath+'\*.jpg').take(160)
  positive = tf.data.Dataset.list_files(posPath+'\*.jpg').take(160)
  negative = tf.data.Dataset.list_files(negPath+'\*.jpg').take(160)
  
  # build a dataset of our data
  positives = tf.data.Dataset.zip((anchor,positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
  negatives = tf.data.Dataset.zip((anchor,negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
  data = positives.concatenate(negatives)
  
  # Build dataloader pipline
  data = data.map(preprocessTwin)
  data = data.cache()
  data = data.shuffle(buffer_size=1024)
  
  # Training partition
  trainData = data.take(round(len(data)*.7))
  trainData = trainData.batch(16)
  trainData = trainData.prefetch(8)
  
  # Testing partition
  testData = data.skip(round(len(data)*.7))
  testData = testData.take(round(len(data)*.3))
  testData = testData.batch(16)
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

class AI:
  def __init__ (self, trainData,testData):
    self.trainData = trainData
    self.testData = testData
  
  
  def makeSiameseModel(self):
    
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
  def trainStep(self, batch):
    
    # Record all of our operations
    with tf.GradientTape() as tape: # Allows for capture of gradients from network
      
      # Get anchor and the positive/negative image
      X = batch[:2]
      # Get label
      y = batch[2]
      
      # Forward pass
      yhat = self.siameseModel(X, training=True)
      # Calculate loss
      loss = self.binaryCrossLoss(y, yhat)
    # print("Model loss is at: ", loss)
    
    # Calculate gradients
    grad = tape.gradient(loss, self.siameseModel.trainable_variables)
    
    # Calculate updated weights and apply to siamese model
    self.opt.apply_gradients(zip(grad, self.siameseModel.trainable_variables))
  
  def train(self, data, EPOCHS):
    # loop through epochs
    for epoch in range(1,EPOCHS+1):
      print('\n Epoch {}/{}'.format(epoch,EPOCHS))
      progbar = tf.keras.utils.Progbar(len(data))
      
      # loop through each batch
      for idx, batch in enumerate(data):
        # Run train steps
        self.trainStep(batch)
        progbar.update(idx+1)
      
      # Save checkpoints
      if epoch % 2 == 0:
        self.checkpoint.save(file_prefix=self.checkpointPrefix)
  
  def trainAI(self, siameseModel):
    self.siameseModel = siameseModel
    # Define loss funktion
    self.binaryCrossLoss = tf.losses.BinaryCrossentropy(from_logits=True)
    # Define optimizer
    self.opt = tf.keras.optimizers.Adam(1e-4) # 0.0001 learningrate
    
    # Establish checkpoints for training
    checkpointDir = './trainingCheckpoints'
    self.checkpointPrefix = os.path.join(checkpointDir, 'ckpt')
    self.checkpoint = tf.train.Checkpoint(opt=self.opt, siameseModel=siameseModel)
    
    # Train the network
    EPOCHS = 4
    self.train(trainData,EPOCHS=EPOCHS)
    self.siameseModel.save('siamesemodel.h5')
    return siameseModel


rewriteDataToMatchNetwork("Niels")

# Get data from files
(trainData, testData) = buildData()

# Build an instance of the class AI
neuralNetwork = AI(trainData=trainData,testData=testData)
print(neuralNetwork.testData)

# Build network
siameseNetwork = neuralNetwork.makeSiameseModel()
siameseNetwork.summary()

# Train network
neuralNetwork.trainAI(siameseModel=siameseNetwork)

# # Reload model 
# siameseModel = tf.keras.models.load_model('siamesemodelv2.h5', 
#                                    custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

# # View model summary
# siameseModel.summary()


# # Get a batch of test data
# testInput, testVal, yTrue = testData.as_numpy_iterator().next()

# # Make predictions
# yHat = siameseNetwork.predict([testInput, testVal])

# # Post processing the results
# results = [1 if prediction > 0.5 else 0 for prediction in yHat]
# print(results)
# print(yTrue)

# # Create a metric object
# m = Recall()
# # Calculating the recall value
# m.update_state(yTrue, yHat)

# # Return recall result
# print("The network got: ",m.result().numpy()*100, "% correkt")

