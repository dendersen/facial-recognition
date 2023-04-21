from SRC.image.imageEditor import sharpen
from SRC.image.imageLoader import loadImgAsArr
from matplotlib import pyplot as plt
from cv2 import cvtColor, COLOR_BGR2RGB

images = []

for i,img in enumerate(loadImgAsArr(100,True,cropOri=True)):
  
  pic = img[0]
  
  # imshow('Cam output: ', pic)
  print(i,end="\r")
  
  pic = sharpen(pic,threshold=5,showSteps=False,strength=1.2,amplification = 1.03)#keep amplification as low as possible!! or the original image will shine through, can be mitigated with threshold
  images.append(cvtColor(pic,COLOR_BGR2RGB))
  # imshow(f'sharp output: ',pic)
  

fig = plt.figure(dpi=300)

size = int(len(images)/2)

for i, image in enumerate(images):
  fig.add_subplot(size,size,i)
  plt.imshow(image, cmap="gray")
  plt.axis("off")
plt.show()

