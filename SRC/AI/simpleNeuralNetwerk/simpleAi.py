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
    
    model = tf.keras.Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu', input_shape=(100, 100, 3)))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(236,activation='relu'))
    model.add(Dropout(rate=.7))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(rate=.7))
    model.add(Dense(units=4))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.summary()
    
    acc,val_acc,loss,val_loss,epochs_range = fitModelToData(model, trainingData,testData,epochs=epochs)
    
    showResults(acc,val_acc,loss,val_loss,epochs_range)
    
    model.save("SimpleAi", save_format='tf')
