from SRC.AI.useAI import AI
import cv2 as cv

a:AI = AI()
escKeyCode = 27
while True:
    a.detectAndDisplayFace()    
    if (cv.waitKey(10) == escKeyCode):
        break
    
