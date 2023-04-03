from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf
import os
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import math
import tarfile

# Avoid out of out of memmory errors by setting GPU Memory Consumption Growth
# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
  tf.config.experimental.set_memory_growth(gpu, True)

def rewriteDataToMatchNetwork(person: str, addFacesInTheWild: bool = False):
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
  
  # Get exstra negative data, from Untar Labelled Faces in the Wild Dataset
  if addFacesInTheWild:
    # Uncompress Tar GZ Labelled faces int the wild
    with tarfile.open('lfw.tgz', "r:gz") as tar:
      # Move LFW Images to the following repository data/negative
      for member in tar.getmembers():
        if member.name.endswith(".jpg") or member.name.endswith(".png"):
          member.name = os.path.basename(member.name)
          tar.extract(member, negPath)
  
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

# Make an new instance of a model
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

def showSiameseBatch(testInput,testVal,yTrue,yHat, person):
  
  inputTmages = testInput
  testImages = testVal
  
  height = math.ceil((len(inputTmages)+len(testImages))/4)
  fig, axs = plt.subplots(4, height, figsize=(16,18))
  indexInput = 0
  indexTest = 0
  fig.suptitle("The person we are looking at is: "+person+" : The first row shows what should be guessed, the second shows the guess", fontsize=14, fontweight='bold')
  for i in range(4):
    for j in range(height):
      if i%2 == 0:
        axs[i,j].imshow(inputTmages[indexInput])
        axs[i,j].set_title(str(yTrue[indexInput]))
        axs[i,j].axis("off")
        indexInput = indexInput+1
      else:
        axs[i,j].imshow(testImages[indexTest])
        axs[i,j].set_title(str(yHat[indexTest][0]))
        axs[i,j].axis("off")
        indexTest = indexTest+1
  plt.show()

class SiameseNeuralNetwork:
  def __init__(self, person: str = "Christoffer", loadAmount: int = 300, trainDataSize: float = 0.7, bachSize: int = 16, addFacesInTheWild: bool = False, resetNetwork: bool = False):
    self.person: str = person
    
    if self.person == "Christoffer":
      self.personName = "Chris"
    else:
      self.personName = self.person
    
    rewriteDataToMatchNetwork(person=self.person, addFacesInTheWild = addFacesInTheWild)
    
    # Optimizer and loss
    self.lossObject = tf.losses.BinaryCrossentropy(from_logits=True)
    self.optimizer = tf.keras.optimizers.Adam(1e-4)
    
    # Get data from files
    (self.trainingData, self.testData) = buildData(loadAmount=loadAmount,trainDataSize=trainDataSize,bachSize=bachSize)
    
    # # Get a batch of test data
    # testInput, testVal, yTrue = testData.as_numpy_iterator().next()
    if resetNetwork:
      self.siameseNetwork = makeSiameseModel()
    else:
      # Reload model 
      self.siameseNetwork = tf.keras.models.load_model("siamesemodel"+self.personName+".h5", 
                                        custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
    
    # Gives a summary of the network
    self.siameseNetwork.summary()
  
  def train(self, EPOCHS: int = 10):
    # Keep results for plotting
    trainLossResults = []
    trainAccuracyResults = []
    testAccuracyResults = []
    
    # loss funktion
    def loss(images, labels, training):
      # training=training is needed only if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      predictions = self.siameseNetwork(images, training=training)
      
      return self.lossObject(y_true=labels, y_pred=predictions)
    
    # gets the gradiants
    @tf.function # Compiles into a tensorflow graph
    def grad(batch: int):
      # Record all of our operations
      with tf.GradientTape() as tape: # Allows for capture of gradients from network
        
        # Get anchor and the positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]
        
        lossValue = loss(images=X,labels=y,training=True)
        
      return lossValue, tape.gradient(lossValue, self.siameseNetwork.trainable_variables)
    
    # checkpointDir = './training_checkpoints'
    # checkpointPrefix = os.path.join(checkpointDir, 'ckpt')
    # checkpoint = tf.train.Checkpoint(opt=self.optimizer, siameseNetwork=self.siameseNetwork)
    
    # loop through epochs
    for epoch in range(EPOCHS):
      
      epochLossAvg = tf.keras.metrics.Mean()
      epochAccuracy = tf.keras.metrics.BinaryAccuracy()
      
      print('\n Epoch {}/{}'.format(epoch,EPOCHS))
      progbar = tf.keras.utils.Progbar(len(self.trainingData))
      
      # # Creating a metric object 
      # r = Recall()
      # p = Precision()
      
      # Loop through each batch
      for idx, batch in enumerate(self.trainingData):
        # Run train step here
        lossValue, grads = grad(batch)
        self.optimizer.apply_gradients(zip(grads, self.siameseNetwork.trainable_variables))
        
        # Track progress
        epochLossAvg.update_state(lossValue)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        X = batch[:2]
        y = batch[2]
        epochAccuracy.update_state(y, self.siameseNetwork(X, training=True))
        
        # Test the mode on the test data
        testAccuracy = tf.keras.metrics.Accuracy()
        
        # training=False is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        logits = self.siameseNetwork(X, training=False)
        prediction = tf.math.argmax(logits, axis=1, output_type=tf.int64)
        testAccuracy(prediction, y)
        progbar.update(idx+1)
      
      # End epoch
      trainLossResults.append(epochLossAvg.result())
      trainAccuracyResults.append(epochAccuracy.result())
      testAccuracyResults.append(testAccuracy.result())
      
      if epoch % 1 == 0:
        # Save checkpoints
        # checkpoint.save(file_prefix=checkpointPrefix)
        
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, Test set accuracy: {:.3%}".format(epoch,
                                                                    epochLossAvg.result(),
                                                                    epochAccuracy.result(),
                                                                    testAccuracy.result()))
    
    # Show how the training went
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')
    
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(trainLossResults)
    
    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(trainAccuracyResults, 'go--', label = 'Train_accuracy')
    axes[1].plot(testAccuracyResults, 'go--', label = 'Test_accuracy' )
    plt.show()
    
    # Replace the old model with the new trained one
    self.siameseNetwork.save('siamesemodel'+self.personName+'.h5')
  
  # Makes some predictions on some data and outputs how sure it was
  def makeAPredictionOnABatch(self):
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
  
  def runSiameseModel(self,Camera, detectionThreshold: float = 0.5, verificationThreshold: float = 0.5):
    """Runs the siamese model you are currently working on
    Args:
        Camera: Takes an instance of the class Cam from imageCapture, e.g Cam(0)
    Returns:
      True if you are verified and false if not
    """
    
    # Initialize a Cam class
    Camera = Camera
    
    while True:
      frame = Camera.readCam()
      
      # Verification trigger
      if cv.waitKey(10) & 0xFF == ord('v'):
        face = Camera.processFace(frame)
        if type(face) == np.ndarray:
          cv.imwrite(os.path.join('images\DataSiameseNetwork', 'inputImages', 'inputImage.jpg'), face)
          # Run verification
          results, verified = verify(self.siameseNetwork, detectionThreshold, verificationThreshold, person=self.person)
          if verified:
            print("This is " + self.person)
            # print("The results are: ", results)
            
          else:
            print("This is not " + self.person)
            # print("The results are: ", results)
            
      
      if cv.waitKey(10) & 0xFF == ord('q'):
        break
    cv.destroyAllWindows()