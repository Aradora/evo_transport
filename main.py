import numpy as np
import random

def mattest(matrix, row_max, col_max):
    row_test = matrix.sum(axis=1,dtype=np.int)
    col_test = matrix.sum(axis=0,dtype=np.int)

    for row in range(0,matrix.shape[0]):
        if row_max[row] != row_test[row]:
            return False
    for col in range(0,matrix.shape[1]):
        if col_max[col] != col_test[col]:
            return False
    return True


def grade(matrix, cost):
    penal = 0
    for row in range(0,cost.shape[0]):
        for col in range(0,cost.shape[1]):
            #print(matrix[row,col], end=" ")
            penal = penal + (matrix[row,col]*cost[row,col])
        #print("\n")
    return penal


def createspeciman(dim,row_max,col_max):
    counter = -1
    while True:
        counter = counter + 1
        matrix = np.zeros(dim, dtype=np.int)
        tmpcol = list(col_max)
        for x in range(0,dim[0]):
            tmprow = row_max[x]
            for y in range(0,dim[1]):
                if tmprow < tmpcol[y]:
                    foobar = tmprow
                else:
                    foobar = tmpcol[y]
                if y == dim[1]-1:
                    randi = tmprow
                else:
                    randi = random.randint(0,foobar)
                matrix[x,y] = randi
                tmprow = tmprow - randi
                tmpcol[y] = tmpcol[y] - randi
                if tmprow < 0:
                    tmprow = 0
                if tmpcol[y] < 0:
                    tmpcol[y] = 0
        if mattest(matrix,row_max,col_max):
            break
    return matrix

#TODO mutacja

#TODO krzyzowanie



cost_mat = np.array([[10,0,20,11 ],[12,7,9,20],[0,14,16,18]]) #macierz kosztow transportu
SUPER_ROZWIAZANE = np.array([[0,5,0,10 ],[0,10,15,0],[5,0,0,0]]) #rozwiazanie modelowe
visited = np.zeros((3,4),dtype=np.int)
sour = 15,25,5
dest = 5,15,15,10

#generacja populacji
Population = []
for i in range(0,40):
    Population.append(createspeciman([3,4],sour,dest))

print("The lesser, the better: \n")
for i in range(0,len(Population)):
    score = grade(Population[i], cost_mat)
    print(score)



