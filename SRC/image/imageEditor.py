# TODO lav en funktion som:
# tager imod et kvadratisk billede af n- størelse af typen nr array 
# billedet har er en buffer rundt om ansigtet

# Skal rykke billedet rundt, men hele ansigtet skal stadig være i frame 
# der skal laves flere forskælige af disse rykkede ansigter

# for værd af disse nye billeder skal der
# laves naturlige lindende versioner af billedet

# sender disse billeder til image saver


import numpy as np

def makeVarients(image: np.ndarray[np.ndarray[np.ndarray[int]]]):
    print(image)
    pass


myArray = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])

makeVarients(myArray)
