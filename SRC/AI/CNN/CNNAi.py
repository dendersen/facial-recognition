import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, MaxPooling3D, Input, Flatten, Dropout
import numpy as np
from random import shuffle
import SRC.image.imageLoader as IL
import SRC.image.imageCapture as IC
from matplotlib import pyplot as plt
import math
import cv2 as cv

from SRC.progBar import progBar

def showBatch(batch):
    classNames = ["Chris","David","Niels","Other"]
    images = batch[0]
    labels = batch[1]
    
    height = math.ceil(len(images)/4)
    fig, axs = plt.subplots(4, height, figsize=(16,18))
    index = 0
    for i in range(4):
        for j in range(height):
            axs[i,j].imshow(images[index])
            axs[i,j].set_title(classNames[labels[index]])
            axs[i,j].axis("off")
            index = index+1
    plt.show()

def fitModelToData(model, trainingData, testData, epochs: int = 5):
    # Keep results for plotting
    trainLossResults = []
    trainAccuracyResults = []
    testAccuracyResults = []
    
    lossObject = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # loss funktion
    @tf.function
    def loss(images, labels, training):
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=training)
        return lossObject(labels, predictions)
    
    # gets the gradiants
    @tf.function # Compiles into a tensorflow graph
    def grad(batch: int):
        # Record all of our operations
        with tf.GradientTape() as tape: # Allows for capture of gradients from network
            
            # Get anchor and the positive/negative image
            X = batch[0]
            # Get label
            y = batch[1]
            
            lossValue = loss(images = X, labels = y,training=True)
            
        return tape.gradient(lossValue, model.trainable_variables), lossValue
    
    # loop through epochs
    for epoch in range(epochs):
        
        epochLossAvg = tf.keras.metrics.Mean()
        epochAccuracy = tf.keras.metrics.BinaryAccuracy()
        testepochAccuracy = tf.keras.metrics.BinaryAccuracy()
        
        print('Epoch {}/{}'.format(epoch+1,epochs))
        progbar = progBar(len(trainingData))
        # Loop through each batch
        for idx, batch in enumerate(trainingData):
            # Run train step here
            grads, lossValue = grad(batch)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            # Track progress
            epochLossAvg.update_state(lossValue)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            X = batch[0]
            y = batch[1]
            epochAccuracy.update_state(y, model(X, training=True))
            # Update progbar
            currentLoss = epochLossAvg.result().numpy()
            currentAccuracy = epochAccuracy.result().numpy()
            progbar.print(idx+1, suffix=f"Loss: {currentLoss:.3f}, Accuracy: {currentAccuracy:.3%}")
        
        for batch in testData:
            X = batch[0]
            y = batch[1]
            # Test the mode on the test data
            # training=False is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            testepochAccuracy.update_state(y, model(X, training=False))
        
        # End epoch
        trainLossResults.append(epochLossAvg.result())
        trainAccuracyResults.append(epochAccuracy.result())
        testAccuracyResults.append(testepochAccuracy.result())
        
        
        print("\nEpoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, Test accuracy: {:.3%}".format(epoch,
                                                                    epochLossAvg.result(),
                                                                    epochAccuracy.result(),
                                                                    testepochAccuracy.result()))
    epochs_range = range(epochs)
    
    return trainAccuracyResults,testAccuracyResults,trainAccuracyResults,epochs_range

def showResults(acc,val_acc,loss,epochs_range):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def makeModel(loadAmount: int, trainDataSize:float = .7, epochs: int = 15):
    print("loading Dataset")
    # 0 = Christoffer, 1 = David, 2 = Niels and 3 = Other
    trainingData, testData = IL.loadDataset(loadAmount,trainDataSize)
    
    # Input
    input_image = Input(shape=(100, 100, 3), name='inputImage')
    
    # First block
    conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='valid', activation='relu')(input_image)
    maxpool1 = MaxPooling2D(pool_size = (2, 2), padding = 'same')(conv1)
    
    # Second block
    conv2 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding="valid", activation="relu")(maxpool1)
    maxpool2 = MaxPooling2D(pool_size = (2, 2), padding = 'same')(conv2)
    
    # Third block
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu")(maxpool2)
    maxpool3 = MaxPooling2D(pool_size = (2, 2), padding = 'same')(conv3)
    
    # First dense block
    flat1 = Flatten()(maxpool3)
    dense1 = Dense(8172, activation='relu')(flat1)
    dropout1 = Dropout(rate=.7)(dense1)
    
    # Second dense block
    dense2 = Dense(4096,activation='relu')(dropout1)
    dropout2 = Dropout(rate=.7)(dense2)
    
    # Output
    dense3 = Dense(4, activation='softmax')(dropout2)
    
    model = tf.keras.Model(inputs=[input_image], outputs=dense3, name='CNN')
    model.summary()
    
    acc,val_acc,loss,epochs_range = fitModelToData(model, trainingData, testData, epochs=epochs)
    
    showResults(acc,val_acc,loss,epochs_range)
    
    model.save("CNNAi", save_format='tf')
