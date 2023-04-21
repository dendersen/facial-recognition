modelSelected = False
selectedModel = 0

print("Welcome to our models. Please select a model: \n",
      "Press 1: KNN \n", 
      "Press 2: Simple Neural Network \n", 
      "Press 3: Siamese Neural Network")

while (not modelSelected):
    answer = input()
    if (answer == "1" or answer == "2" or answer == "3"):
        print("Model " + answer + " was selected")
        modelSelected = True
        selectedModel = int(answer)
    else:
        print("Please select a correct model")
    pass    

if(selectedModel == 1):
    print("Do cool knn ting")
    
if(selectedModel == 2):
    print("Do cool Simple Neural Network ting")
    
if(selectedModel == 3):
    print("Do cool Siamese Neural Network ting")