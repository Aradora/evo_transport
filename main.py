import numpy as np
import random
from matplotlib.pyplot import plot,show,figure,title,legend,ylabel,xlabel


def verify(matrix, row_max, col_max):
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


def createspeciman(dimension,row_max,col_max):
    counter = -1
    while True:
        counter = counter + 1
        matrix = np.zeros(dimension, dtype=np.int)
        tmpcol = list(col_max)
        for x in range(0,dimension[0]):
            tmprow = row_max[x]
            for y in range(0,dimension[1]):
                if tmprow < tmpcol[y]:
                    foobar = tmprow
                else:
                    foobar = tmpcol[y]
                if y == dimension[1]-1:
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
        if verify(matrix,row_max,col_max):
            break
    return matrix


#selekcja turniejowa
#TODO: fix percentage option
def tournament(population, tournamentsize, percentage, basepopulation, cost): #percentage is flawed and errors the program
    popbis = []
    changedpop = int(np.floor(basepopulation*(percentage/100)))
    while len(popbis) < changedpop:
        competitors =[]
        score = []
        for i in range(0,tournamentsize):
            competitors.append(population[random.randint(1,len(population))-1])
        for i in range(0,len(competitors)):
            score.append(grade(competitors[i],cost))
        popbis.append(competitors[score.index(min(score))])
        for j in range(0,len(popbis)-changedpop):
            popbis.append(population[random.randint(1,len(population))-1])
    return popbis

#TODO: add roulette selection

#krzyzowanie
def crossover(parent1,parent2):
    DIV = np.zeros(parent1.shape, dtype=np.int)
    REM = np.zeros(parent1.shape, dtype=np.int)
    REM1 = np.zeros(parent1.shape, dtype=np.int)
    REM2 = np.zeros(parent1.shape, dtype=np.int)
    for x in range(0,parent1.shape[0]):
        for y in range(0, parent1.shape[1]):
            DIV[x,y] = int(np.floor(parent1[x,y] + parent2[x,y])/2)
            REM[x,y] = int(parent1[x,y]+parent2[x,y])%2

    for x in range(0,REM.shape[0]):
        for y in range(0, REM.shape[1]):
            if REM[x,y] == 1:
                REM[x, y] = 0
                REM1[x,y] = 1
                for k in range(i,REM.shape[0]):
                    if REM[k,y] == 1:
                        REM[k,y] = 0
                        REM2[k,y] = 1
                        break
                for m in range(y,REM.shape[1]):
                    if REM[x,m] == 1:
                        REM[x,m] = 0
                        REM2[x,m] = 1
                        break
    return DIV+REM1,DIV+REM2


def mutation(matrix, dimension):
    rand_x_l = random.randint(0,dimension[0]-2)
    rand_x_h = random.randint(rand_x_l+1,dimension[0]-1)
    rand_y_l = random.randint(0,dimension[1]-2)
    rand_y_h = random.randint(rand_y_l+1,dimension[1]-1)
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

    new_dimension = [rand_x_h-rand_x_l+1, rand_y_h-rand_y_l+1]
    #print(new_dimension)
    new_matrix = createspeciman(new_dimension,row_max,col_max)
    #print(new_matrix)

    #print("Stara macierz:")
    #print(matrix)

    for x in range(0, new_dimension[0]):
        for y in range(0, new_dimension[1]):
            matrix[x+rand_x_l,y+rand_y_l] = new_matrix[x,y]
    #print("Zwracamy nowa:")
    #print(matrix)
    return matrix

cost_mat = np.array([[10,0,20,11 ],[12,7,9,20],[0,14,16,18]]) #macierz kosztow transportu
SUPER_ROZWIAZANE = np.array([[0,5,0,10 ],[0,10,15,0],[5,0,0,0]]) #rozwiazanie modelowe
sour = 15,25,5 #max w wierszach
dest = 5,15,15,10 #max w kolumnach
dimension = [3,4]

#
#alternatywnie:
#cost_mat = np.array([[5,3,1,2,2 ],[2,1,1,1,1],[1,1,2,5,2],[5,4,3,1,6]]) #macierz kosztow transportu
#SUPER_ROZWIAZANE = np.array([[0,0,0,0,20 ],[0,15,0,0,15],[10,0,0,0,0],[0,0,30,10,0]]) #rozwiazanie modelowe
#sour = 20,30,10,40 #max w wierszach
#dest = 10,15,30,10,35 #max w kolumnach
#dimension = [4,5]
#

GENERATIONS = 100
POP_SIZE = 100
competitors = 4
percentageofchanged = 100
mutationchance = 100

simscore = []
history = []
genscore = []
avgsimscore = []


simulations = 1
for simulation in range(0, simulations):
    simscore.clear()
    avgsimscore.clear()
    # generacja populacji
    Population = []
    for i in range(0, POP_SIZE):
        Population.append(createspeciman(dimension, sour, dest))
    pairs = len(Population)/2

    #metoda turniejowa
    for i in range(0,GENERATIONS):
        for j in range(0,int(pairs)):
            child1,child2 = crossover(Population[j],Population[len(Population)-j-1])
            if verify(child1,sour,dest):
                Population.append(child1)
            if verify(child2, sour, dest):
                Population.append(child2)
        for individual in range(0,POP_SIZE):
            if random.randint(0,mutationchance) == 28:
                Population[individual] = mutation(Population[individual],dimension)
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
xlabel('nr symulacji')
ylabel('wynik')
figure()


plot(simscore,'-b',avgsimscore,'-c',[0, GENERATIONS],targetline,'--r')
title("przebieg jednej symulacji")
legend([" przystosowanie najlepszego osobnika "," srednie przystosowanie osobnikow"," najlepszy wynik "])
xlabel('pokolenie')
ylabel('wynik')
show()