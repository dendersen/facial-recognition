def isFloat(string):
    if string.replace(".", "").isnumeric():
        return True
    else:
        return False

def chooseModel() -> int:
    answer = input()
    if (answer == "1" or answer == "2"):
        print("Model " + answer + " was selected")
        return int(answer)
    else:
        print("Please select a correct model")
        chooseModel()
        
def chooseDataSize():
    answer = input()
    if (answer.isdigit() and int(answer) > 0):
        print("Data size of " + answer + " was selected")
        return int(answer)
    else:
        print("Please select a valid amount (int > 0)")
        chooseDataSize()

def chooseTrainingSize():
    answer = input()
    if (isFloat(answer) and float(answer)< 1 and float(answer)> 0):
        print("training size of " + answer + " was selected")
        return float(answer)
    else:
        print("Please select a valid amount (between 1 and 0)")
        chooseTrainingSize()
        
def chooseEpochs():
    answer = input()
    if (answer.isdigit() and int(answer) > 0):
        print("Epoch size of " + answer + " was selected")
        return int(answer)
    else:
        print("Please select a valid amount (int > 0)")
        chooseEpochs() 

print("Welcome to our models. Please select a model to train: \n",
    "Press 1: Simple Neural Network \n", 
    "Press 2: Siamese Neural Network")

modelNumber = chooseModel()

print("please select the data size:")
loadAmount = chooseDataSize()

print("please select how much of the data should be used for training. standard is 0.7:")
trainingSize = chooseTrainingSize()

print("How many epochs should the model be trained on:")
epochAmount = chooseEpochs()

if(modelNumber == 1):
    print("importing")
    import SRC.AI.simpleNeuralNetwerk.simpleAi as simpleNetwork
    simpleNetwork.makeModel(loadAmount,trainingSize,epochAmount)
    
if(modelNumber == 2):
    print("not fully implemented yet")