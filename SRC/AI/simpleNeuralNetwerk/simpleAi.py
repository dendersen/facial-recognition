import tensorflow as tf
import numpy as np
from random import shuffle
import SRC.image.imageLoader as IL

print("hello world")

# # how big the train data should. 1 is the whole dataset 0.5 is half
# trainDataSize = 0.6

# # load data
# data = IL.loadImgAsArr(30,True,alowModified=True)    
# shuffle(data)

# # seperate into traning and test
# firstTestIndex = round(len(data)*trainDataSize)

# traningImages = []
# traningLabels = []
# for i,image in zip(range(0,firstTestIndex),data):
#     traningImages.append(image[0])
#     traningLabels.append(image[1])

# testImages = []
# testLabels = []
# for i,image in zip(range(firstTestIndex,len(data)),data):
#     testImages.append(image[0])
#     testLabels.append(image[1])

# 0 = Christoffer, 1 = David, 2 = Niels and 3 = Other
traningData, testData = IL.loadDataset(loadAmount=32,trainDataSize=.7)

print("Traning Labels: " + str(traningData[:][1]))
print("Testing Labels: " + str(testData[:][1]))




