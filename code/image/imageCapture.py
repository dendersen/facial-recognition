import cv2 as cv
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--faceCascade', help='Path to face cascade.', default='code\image\haarcascade_frontalface_alt.xml')
parser.add_argument('--eyesCascade', help='Path to eyes cascade.', default='code\image\haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()

faceCascadeName = args.faceCascade
eyesCascadeName = args.eyesCascade
faceCascade = cv.CascadeClassifier()
eyesCascade = cv.CascadeClassifier()

#-- 1. Load the cascades
if not faceCascade.load(cv.samples.findFile(faceCascadeName)):
  print('--(!)Error loading face cascade')
  exit(0)
if not eyesCascade.load(cv.samples.findFile(eyesCascadeName)):
    print('--(!)Error loading eyes cascade')
    exit(0)

def detectAndDisplayFace(frame):
  frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  frameGray = cv.equalizeHist(frameGray)
  #-- Detect faces
  faces = faceCascade.detectMultiScale(frameGray)
  print("Found {0} faces!".format(len(faces)))
  for (x,y,w,h) in faces:
    center = (x + w//2, y + h//2)
    frame = cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    faceROI = frameGray[y:y+h,x:x+w]
    #-- In each face, detect eyes
    eyes = eyesCascade.detectMultiScale(faceROI)
    for (x2,y2,w2,h2) in eyes:
      eyeCenter = (x + x2 + w2//2, y + y2 + h2//2)
      radius = int(round((w2 + h2)*0.25))
      frame = cv.circle(frame, eyeCenter, radius, (255, 0, 0 ), 4)
  cv.imshow('Capture - Face detection', frame)

# img = cv.imread('images\\original\\Christoffer\\3.jpg')
# scale_percent = 30 # percent of original size
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
# img = cv.resize(img,dim, interpolation= cv.INTER_AREA)
# detectAndDisplayFace(frame=img)
# cv.waitKey(0)

camera_device = args.camera
#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
  print('--(!)Error opening video capture')
  exit(0)

while True:
  ret, frame = cap.read()
  if frame is None:
    print('--(!) No captured frame -- Break!')
    break
  detectAndDisplayFace(frame)
  if cv.waitKey(10) == 27:
      break