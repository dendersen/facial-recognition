import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

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

  def trainAI():
    # Define loss funktion
    binaryCrossLoss = tf.losses.BinaryCrossentropy(from_logits=True)
    
    # Define optimizer
    opt = tf.keras.optimizers.Adam
    
  # # Load the data
  # trainData = keras.utils.image_dataset_from_directory('images\modified',
  #                                                         shuffle=True,
  #                                                         image_size=(100,100))
  
  pass




siameseNetwork = AI.makeSiameseModel()
siameseNetwork.summary()