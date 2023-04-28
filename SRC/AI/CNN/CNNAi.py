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
    history = model.fit(
        trainingData,
        validation_data=testData,
        epochs=epochs
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    
    return acc,val_acc,loss,val_loss,epochs_range

def showResults(acc,val_acc,loss,val_loss,epochs_range):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
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
    dense3 = Dense(4, activation='sigmoid')(dropout2)
    
    model = tf.keras.Model(inputs=[input_image], outputs=[dense3], name='CNN')
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
    model.summary()
    
    acc,val_acc,loss,val_loss,epochs_range = fitModelToData(model, trainingData,testData,epochs=epochs)
    
    showResults(acc,val_acc,loss,val_loss,epochs_range)
    
    model.save("CNNAi", save_format='tf')
