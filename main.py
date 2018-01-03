import numpy as np
import random
from matplotlib.pyplot import plot
from matplotlib.pyplot import show

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


#selekcja turniejowa
def tournament(population, toursize, lasted, cost):
    popbis = []
    while len(popbis) < lasted :
        competitors =[]
        score = []
        for i in range(0,toursize):
            competitors.append(population[random.randint(1,len(population))-1])
        for i in range(0,len(competitors)):
            score.append(grade(competitors[i],cost))
        popbis.append(competitors[score.index(min(score))])
    return popbis

#krzyzowanie
def crossover(parent1,parent2):
    DIV = np.zeros(parent1.shape, dtype=np.int)
    REM = np.zeros(parent1.shape, dtype=np.int)
    REM1 = np.zeros(parent1.shape, dtype=np.int)
    REM2 = np.zeros(parent1.shape, dtype=np.int)
    for i in range(0,parent1.shape[0]):
        for j in range(0, parent1.shape[1]):
            DIV[i,j] = int(np.floor(parent1[i,j] + parent2[i,j])/2)
            REM[i,j] = int(parent1[i,j]+parent2[i,j])%2

    REMbis = REM.copy()
    for i in range(0,REMbis.shape[0]):
        for j in range(0, REMbis.shape[1]):
            if REMbis[i,j] == 1:
                REMbis[i, j] = 0
                REM1[i,j] = 1
                for k in range(i,REMbis.shape[0]):
                    if REMbis[k,j] == 1:
                        REMbis[k,j] = 0
                        REM2[k,j] = 1
                        break
                for m in range(j,REMbis.shape[1]):
                    if REMbis[i,m] == 1:
                        REMbis[i,m] = 0
                        REM2[i,m] = 1
                        break



    return DIV+REM1,DIV+REM2


#TODO mutacja


cost_mat = np.array([[10,0,20,11 ],[12,7,9,20],[0,14,16,18]]) #macierz kosztow transportu
SUPER_ROZWIAZANE = np.array([[0,5,0,10 ],[0,10,15,0],[5,0,0,0]]) #rozwiazanie modelowe
sour = 15,25,5 #max w wierszach
dest = 5,15,15,10 #max w kolumnach
GENERATIONS = 30
POP_SIZE = 600
score = []
history = []

#generacja populacji
Population = []
for i in range(0,POP_SIZE):
    Population.append(createspeciman([3,4],sour,dest))

for z in range(0,10):
    #metoda turniejowa
    for i in range(0,GENERATIONS):
        genscore = []
        for j in range(0,int(len(Population)/2)):
            child1,child2 = crossover(Population[j],Population[len(Population)-j-1])
            Population.append(child1)
            Population.append(child2)
        Population = tournament(Population, 4, 40, cost_mat)
        for i in range(0, len(Population)):
            genscore.append(grade(Population[i], cost_mat))
        score.append(min(genscore))


print("The lesser, the better: \n")
targetline = [grade(SUPER_ROZWIAZANE,cost_mat),grade(SUPER_ROZWIAZANE,cost_mat)]
plot(score,'-b',[0,GENERATIONS],targetline,'-r')
show()
print(np.min(score))
print(grade(SUPER_ROZWIAZANE,cost_mat))


