# from SRC.image.imageCapture import Cam
# from SRC.image.imageEditor import *
import SRC.image.imageLoader as IL
# import SRC.image.imageSaver as IS
# import cv2 as cv

IL.modifyOriginals()


# # make an instance of camera
# Camera = Cam(0)

# while True:
#   pic = Camera.readCam()
#   if cv.waitKey(10) == 32:
#     BGRface = Camera.processFace(pic)
#     if type(BGRface) == np.ndarray:
#         # save original face
#         RGBface = cv.cvtColor(BGRface, cv.COLOR_BGR2RGB)
#         print("This is the shape of the face picture: ", RGBface.shape)
#         IS.saveImage([RGBface],"Christoffer",False)
        
#         # make variant
#         BGRnewVariants = makeVarients(BGRface)
        
#         # save all variants
#         for variant in BGRnewVariants:
#           # change from BGR to RGB
#           variant = cv.cvtColor(variant, cv.COLOR_BGR2RGB)
#           print("This is the shape of the variant: ", variant.shape)
#           # save variant
#           IS.saveImage([variant],"Christoffer",True)
#   if cv.waitKey(10) == 27:
#     break
