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

def chooseLabel():
    print("Please select a person: Christoffer, David or Niels")
    answer = input()
    if (answer == "Christoffer" or answer == "David" or answer == "Niels"):
        print("Label " + answer + " was selected")
        return answer
    else:
        print("Please select a valid label: Christoffer, David or Niels")
        chooseLabel()

def chooseIfNetworkShouldReset():
    print("Please select if the network should be reset: True or False")
    answer = input()
    if (answer == "True"):
        print("You chose " + answer)
        return True
    if (answer == "False"):
        print("You chose " + answer)
        return False
    else:
        print("Please select a valid option: True or False")
        chooseIfNetworkShouldReset()

def chooseIfNetworkShouldLoadNewData():
    print("Please select if the network should load new data: True or False")
    answer = input()
    if (answer == "True"):
        print("You chose " + answer)
        return True
    if (answer == "False"):
        print("You chose " + answer)
        return False
    else:
        print("Please select a valid option: True or False")
        chooseIfNetworkShouldLoadNewData()

def chooseVariants():
    print("please select the amount of variants:")
    answer = input()
    if (answer.isdigit() and int(answer) > 0 and int(answer) < 10):
        print("Epoch size of " + answer + " was selected")
        return int(answer)
    else:
        print("Please select a valid amount (0 < int > 10)")
        chooseVariants() 

def chooseIfReprocessDataset():
    print("Please select if the network reload the data from Labels In The Wild: True or False")
    answer = input()
    if (answer == "True"):
        print("You chose " + answer)
        return True
    if (answer == "False"):
        print("You chose " + answer)
        return False
    else:
        print("Please select a valid option: True or False")
        chooseIfReprocessDataset()

def chooseIfYouShouldUseDataset():
    print("Please select if the network should use the dataset: True or False")
    answer = input()
    if (answer == "True"):
        print("You chose " + answer)
        return True
    if (answer == "False"):
        print("You chose " + answer)
        return False
    else:
        print("Please select a valid option: True or False")
        chooseIfYouShouldUseDataset()

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
    print("Importing")
    import SRC.AI.simpleNeuralNetwerk.simpleAi as simpleNetwork
    simpleNetwork.makeModel(loadAmount,trainingSize,epochAmount)

if(modelNumber == 2):
    print("Importing")
    from trainSiameseNetwork import trainAndEvaluate
    label = chooseLabel()
    resetNetwork = chooseIfNetworkShouldReset()
    loadNewData = chooseIfNetworkShouldLoadNewData()
    variantAmount = chooseVariants()
    reprocessDataset = chooseIfReprocessDataset()
    useDataset = chooseIfYouShouldUseDataset()
    trainAndEvaluate(loadNewData = loadNewData, loadAmount = loadAmount, varients = variantAmount, trainDataSize = trainingSize, EPOCHS = epochAmount, label = label, reprocessDataset = reprocessDataset)