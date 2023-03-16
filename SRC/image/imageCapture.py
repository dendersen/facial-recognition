import cv2 as cv

# img = cv.imread('images\\original\\Christoffer\\3.jpg')
# scale_percent = 30 # percent of original size
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
# img = cv.resize(img,dim, interpolation= cv.INTER_AREA)
# detectAndDisplayFace(frame=img)
# cv.waitKey(0)
class Cam:
  def __init__(self,cameraDevice) -> None:
    self.cameraDevice = cv.VideoCapture(cameraDevice)
    pass
  def readCam(self):
    #-- 2. Read the video stream
    if not self.cameraDevice.isOpened:
      print('--(!)Error opening video capture')
      exit(0)
    ret, frame = self.cameraDevice.read()
    if frame is None:
      print('--(!) No captured frame -- Break!')
      return
    return frame