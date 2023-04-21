import math
from SRC.image.imageEditor import sharpen, printProgressBar
from SRC.image.imageLoader import loadImgAsArr
from matplotlib import pyplot as plt
from time import time

images = []
tim  = time()
allItems = loadImgAsArr(10,False,cropOri=True)
for i,img in enumerate(allItems):
  pic = img[0]
  pic = sharpen(pic,threshold=5,showSteps=False,strength=1.2,amplification = 1.03)#keep amplification as low as possible!! or the original image will shine through, can be mitigated with threshold
  printProgressBar(i,len(allItems),tim)
  images.append(pic)
print("\r" + (" "*100) + "\rdone")

# size = int(math.ceil(math.sqrt(len(images))))
for i, image in enumerate(zip(images,allItems)):
  fig = plt.figure(dpi=300)
  fig.add_subplot(2,1,1)
  plt.imshow(image[0], cmap="gray")  
  plt.axis("off")
  fig.add_subplot(2,1,2)
  plt.imshow(image[1][0], cmap="gray")
  plt.axis("off")
  plt.show()

