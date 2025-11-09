# Creare inizialmente una matrice L 4 x 4 tutta nulla ma la cui diagonale sia
# -1, 1, 1, 1 e poi creare una matrice B definita come 
# B = [[gamma, - beta * gamma, 0, 0],
#      [-beta*gamma, gamma, 0, 0],
#      [0, 0, 1, 0],
#       [0, 0, 0, 1]]
import numpy as np
L = np.zeros((4, 4))

idx = [0, 1, 2, 3]
# L[i][j] oppure L[i, j]
L[idx, idx] = [1, -1, -1, -1]
print(L)
beta = 0.1
gamma = 1/np.sqrt(1-beta**2)
B = np.zeros((4, 4))
B[idx, idx] = [gamma, gamma, 1, 1]
B[1, 0] = B[0, 1] = - gamma * beta
print(B)