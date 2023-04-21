from SRC.image.imageEditor import sharpen, printProgressBar
from SRC.image.imageLoader import loadImgAsArr
from matplotlib import pyplot as plt
from cv2 import cvtColor, COLOR_BGR2RGB
from time import time

images = []
tim  = time()
allItems = loadImgAsArr(10,True,cropOri=True)
for i,img in enumerate(allItems):
  pic = img[0]
  pic = sharpen(pic,threshold=5,showSteps=False,strength=1.2,amplification = 1.03)#keep amplification as low as possible!! or the original image will shine through, can be mitigated with threshold
  printProgressBar(i+1,len(allItems),tim)

fig = plt.figure(dpi=300)

size = int(len(images)/2)

for i, image in enumerate(images):
  fig.add_subplot(size,size,i+1)
  plt.imshow(image, cmap="gray")
  plt.axis("off")
plt.show()

