import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import os
import glob
import cv2
from matplotlib import pyplot as plt

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
  
  return data

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
    
    # Build the training step - This is where i left off from
  pass


rewriteDataToMatchNetwork("Christoffer")

data = buildData()
samples = data.as_numpy_iterator()
print(len(samples.next()))
fig, ax = plt.subplots(2)
ax[0].imshow(samples.next()[0])
ax[1].imshow(samples.next()[1])
plt.show()
print(samples.next()[2])

# siameseNetwork = AI.makeSiameseModel()
# siameseNetwork.summary()

