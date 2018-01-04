import numpy as np
import random
from matplotlib.pyplot import plot,show,figure,title,legend


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
def tournament(population, toursize, percentage, basepopulation, cost):
    popbis = []
    changedpop = int(np.floor(basepopulation*(percentage/100)))
    while len(popbis) < changedpop:
        competitors =[]
        score = []
        for i in range(0,toursize):
            competitors.append(population[random.randint(1,len(population))-1])
        for i in range(0,len(competitors)):
            score.append(grade(competitors[i],cost))
        popbis.append(competitors[score.index(min(score))])
        for j in range(0,len(popbis)-changedpop):
            popbis.append(population[random.randint(1,len(population))-1])
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
def mutation(matrix, dim):
    rand_x_l = random.randint(0,dim[0]-2)
    rand_x_h = random.randint(rand_x_l+1,dim[0]-1)
    rand_y_l = random.randint(0,dim[1]-2)
    rand_y_h = random.randint(rand_y_l+1,dim[1]-1)
    #print("Mutacja nastepuje w x od " +str(rand_x_l)+ " do " +str(rand_x_h)+ " i y od " +str(rand_y_l)+ " do " +str(rand_y_h))

    row_max = []
    for x in range(rand_x_l,rand_x_h+1):
        buf_row = 0
        for y in range(rand_y_l,rand_y_h+1):
            buf_row += matrix[x,y]
        row_max.append(buf_row)
    #print(row_max)

    col_max = []
    for y in range(rand_y_l,rand_y_h+1):
        buf_col = 0
        for x in range(rand_x_l,rand_x_h+1):
            buf_col += matrix[x,y]
        col_max.append(buf_col)
    #print(col_max)

    new_dim = [rand_x_h-rand_x_l+1, rand_y_h-rand_y_l+1]
    #print(new_dim)
    new_matrix = createspeciman(new_dim,row_max,col_max)
    #print(new_matrix)

    #print("Stara macierz:")
    #print(matrix)

    for x in range(0, new_dim[0]):
        for y in range(0, new_dim[1]):
            matrix[x+rand_x_l,y+rand_y_l] = new_matrix[x,y]
    #print("Zwracamy nowa:")
    #print(matrix)
    return matrix

cost_mat = np.array([[10,0,20,11 ],[12,7,9,20],[0,14,16,18]]) #macierz kosztow transportu
SUPER_ROZWIAZANE = np.array([[0,5,0,10 ],[0,10,15,0],[5,0,0,0]]) #rozwiazanie modelowe
sour = 15,25,5 #max w wierszach
dest = 5,15,15,10 #max w kolumnach
dimen = [3,4]
#
#alternatywnie:
#cost_mat = np.array([[5,3,1,2,2 ],[2,1,1,1,1],[1,1,2,5,2],[5,4,3,1,6]]) #macierz kosztow transportu
#SUPER_ROZWIAZANE = np.array([[0,0,0,0,20 ],[0,15,0,0,15],[10,0,0,0,0],[0,0,30,10,0]]) #rozwiazanie modelowe
#sour = 20,30,10,40 #max w wierszach
#dest = 10,15,30,10,35 #max w kolumnach
#dimen = [4,5]
#
GENERATIONS = 40
POP_SIZE = 600
competitors = 4
percentageofchanged = 100

simscore = []
history = []
genscore = []
avgsimscore = []


simulations = 1
for z in range(0, simulations):
    # generacja populacji
    Population = []
    for i in range(0, POP_SIZE):
        Population.append(createspeciman(dimen, sour, dest))
    pairs = len(Population)/2
    #metoda turniejowa

    simscore.clear()
    avgsimscore.clear()
    for i in range(0,GENERATIONS):
        for j in range(0,int(pairs)):
            child1,child2 = crossover(Population[j],Population[len(Population)-j-1])
            if mattest(child1,sour,dest):
                Population.append(child1)
            if mattest(child2, sour, dest):
                Population.append(child2)
        for individual in range(0,POP_SIZE):
            if random.randint(0,999) == 28:
                Population[individual] = mutation(Population[individual],dimen)
        Population = tournament(Population, competitors, percentageofchanged,POP_SIZE, cost_mat)

        for i in range(0, len(Population)):
            genscore.append(grade(Population[i], cost_mat))
        avgsimscore.append(sum(genscore)/POP_SIZE)
        simscore.append(min(genscore))
        genscore.clear()

    print(Population[0])
    history.append(min(simscore))


targetline = [grade(SUPER_ROZWIAZANE,cost_mat),grade(SUPER_ROZWIAZANE,cost_mat)]
#print("The lesser, the better: \n")
for simulation in range(0,simulations):
    plot(simulation,history[simulation],'bD',[0,simulations],targetline,'r--')
title("wynik przeprowadzonych symulacji")
figure()


plot(simscore,'-b',avgsimscore,'-c',[0, GENERATIONS],targetline,'--r')
title("przebieg jednej symulacji")
legend([" przystosowanie najlepszego osobnika "," srednie przystosowanie osobnikow"," najlepszy wynik "])
show()