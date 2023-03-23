import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import os
import glob
import cv2

def rewriteDataToMatchNetwork(person):
  posPath = os.path.join('images\DataSiameseNetwork','positive')
  for f in files:
    os.remove(f)
  negPath = os.path.join('images\DataSiameseNetwork','negative')
  for f in files:
    os.remove(f)
  ancPath = os.path.join('images\DataSiameseNetwork','anchor')
  for f in files:
    os.remove(f)
  
  
  # Get all negative data
  for picture in os.listdir('images\modified\Other'):
    if ".jpg" in picture:
      path = os.path.join('images\modified\Other', picture)
      img = cv2.imread(path)
      newPath = os.path.join(negPath, picture)
      cv2.imwrite(newPath, img)
  
  # Get all anchor data
  datapath = os.path.join('images/original/', person)
  for picture in os.listdir(datapath):
    if ".jpg" in picture:
      path = os.path.join(datapath, picture)
      img = cv2.imread(path)
      newPath = os.path.join(ancPath, picture)
      cv2.imwrite(newPath, img)
  
  # Get all positive data
  datapath2 = os.path.join('images/modified/', person)
  for picture in os.listdir(datapath2):
    if ".jpg" in picture:
      path = os.path.join(datapath2, picture)
      img = cv2.imread(path)
      newPath = os.path.join(posPath, picture)
      cv2.imwrite(newPath, img)

def buildData():
  
  pass

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

@tf.function # Compiles into a tensorflow graph
def trainStep(batch):
  
  with tf.GradientTape() as tape: # Allows for capture of gradients from network
    
    # Get anchor and the positive/negative image
    X = batch[:2]
    # Get label
    y = batch[2]
  
  pass

class L1Dist(Layer):
  
  def __init__(self, **kwargs):
    super().__init__()
  
  # compare embeddings - similarity calculation
  def call(self, inputEmbedding, validationEmbedding):
    return tf.math.abs(inputEmbedding - validationEmbedding)

class AI:
  
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

  def trainAI(siameseModel):
    # Define loss funktion
    binaryCrossLoss = tf.losses.BinaryCrossentropy(from_logits=True)
    # Define optimizer
    opt = tf.keras.optimizers.Adam(1e-4) # 0.0001 learningrate
    
    # Establish checkpoints for training
    checkpointDir = './trainingCheckpoints'
    checkpointPrefix = os.path.join(checkpointDir, 'ckpt')
    checkpoint = tf.train.Checkpoint(opt=opt, siameseModel=siameseModel)
    
    # Build the training step
  
  # # Load the data
  # trainData = keras.utils.image_dataset_from_directory('images\modified',
  #                                                         shuffle=True,
  #                                                         image_size=(100,100))
  
  pass


rewriteDataToMatchNetwork("Niels")

# siameseNetwork = AI.makeSiameseModel()
# siameseNetwork.summary()

