# import SRC.AI.simpleNeuralNetwerk.simpleAi as simpleNeuralNetwork
from SRC.image.imageCapture import Cam
from SRC.AI.siameseAI import *

# Optimizer and loss
binaryCrossLoss = tf.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(1e-4)

rewriteDataToMatchNetwork("Christoffer")

# Get data from files
(trainingData, testData) = buildData()

# Get a batch of test data
testInput, testVal, yTrue = testData.as_numpy_iterator().next()

# Reload model 
siameseNetwork = tf.keras.models.load_model('siamesemodelChris.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

# # Make an new instance of a model
# siameseNetwork = makeSiameseModel()
siameseNetwork.summary()

# # Train model if needed
# train(siameseNetwork=siameseNetwork,data=trainingData,EPOCHS=50, optimizer=optimizer, binaryCrossLoss=binaryCrossLoss)
# makePrediction(siameseNetwork=siameseNetwork, testInput=testInput,testVal=testVal,yTrue=yTrue)

# # Save weights
# siameseNetwork.save('siamesemodelChris.h5')

Cam = Cam(0)
while True:
  frame = Cam.readCam()
  
  # Verification trigger
  if cv.waitKey(10) & 0xFF == ord('v'):
    face = Cam.processFace(frame)
    if type(face) == np.ndarray:
      cv.imwrite(os.path.join('images\DataSiameseNetwork', 'inputImages', 'inputImage.jpg'), face)
      
      person: str = "Christoffer"
      # Run verification
      results, verified = verify(siameseNetwork, 0.5, 0.5, person=person)
      if verified:
        print("This is " + person)
      else:
        print("This is not " + person)
  
  if cv.waitKey(10) & 0xFF == ord('q'):
    break
cv.destroyAllWindows()