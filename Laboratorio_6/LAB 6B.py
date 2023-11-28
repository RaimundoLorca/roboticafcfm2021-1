#LAB 6 - Robótica 2021-1 - ALFORITMOS GENETICOS

#Cargar librerias a usar
import random 
import numpy as np
import matplotlib.pyplot as plot

# Función objetivo:  
def funcion_objetivo(x,y):
    
    r = np.sqrt((x**2)+(y**2)) 
    z = (1/(1+(r**0.7)))*np.cos(y/2.5)*np.sin(x/2.5)
       
    return z + 0.3
    
#%% Definición de la funciónes necesarias para el algoritmo

def Init(n_min,n_max):
    
    indivX = random.uniform(n_min, n_max)
    indivY = random.uniform(n_min, n_max)
    
    return np.array([indivX, indivY])

def Cross(indiv_1, indiv_2, prob = 0.5):
    
    fathers = [indiv_1, indiv_2]
    new_indiv = random.choice(fathers)
    
    if random.random() < prob:
        new_indiv = (indiv_1 + indiv_2)/2  
        
    return new_indiv
      
def Mutator(indiv, n_min, n_max, prob = 0.1):
    
    new_indiv = indiv
    
    if random.random() < prob:
        new_indiv = Init(n_min,n_max)
        
    return new_indiv

def Eval_fitness(indiv):
    
    score = funcion_objetivo(indiv[0],indiv[1])
    
    return score

def Selector(population, k=15):
    
    tournament = random.sample(population, k)
    
    scores = []
    
    for indiv in tournament:
        
        score_indiv = Eval_fitness(indiv)
        scores.append(score_indiv)
         
    index_max = np.argmax(scores)
        
    best_fittnes = scores[index_max]
    best_indiv = tournament[index_max]
    
    tournament.pop(index_max)
    others = tournament
    
    return best_indiv , others

def Best(pob):
    
    scores = []
    
    for indiv in pob:
        
        score_indiv = Eval_fitness(indiv)
        scores.append(score_indiv)
         
    index_max = np.argmax(scores)
        
    best_fittnes = scores[index_max]
    best_indiv = pob[index_max]
    
    return best_fittnes, best_indiv

#%% Armar la estructura del algoritmo

def Fitness_calculator(prob_cross, prob_mut, pop_num):
    
    n_min = -20
    n_max = 20
    
    gen_iter = 600
    population = [Init(n_min, n_max) for i in range(pop_num)]
    
    Fitness = []
    
    for i in range(gen_iter):
    
        new_population = []
        
        while len(new_population) < pop_num:
            
            #seleccion
            best_indiv, rest_tournament = Selector(population)
            
            #cruzamiento
            for indiv in rest_tournament:
                
                new_indiv = Cross(best_indiv, indiv, prob = prob_cross)
                
                new_indiv = Mutator(new_indiv, n_min, n_max, prob=prob_mut)
                
                new_population.append(new_indiv)
                
                if len(new_population) == pop_num:
                    break
    
        population = new_population
    
        best_fitness, best_indiv = Best(population)
        
        Fitness.append(best_fitness)
    
    return Fitness
    
#%% Dependencia de la probabilidad de cruzamiento  

C1 = Fitness_calculator(0.05, 0.10, 20)

C2 = Fitness_calculator(0.3, 0.10, 20)

C3 = Fitness_calculator(0.5, 0.10, 20)

C4 = Fitness_calculator(0.9, 0.10, 20)

plot.figure(1)
plot.plot(C1, label="Prob_cross = 5%")
plot.plot(C2, label='Prob_cross = 30%')
plot.plot(C3, label='Prob_cross = 50%')
plot.plot(C4, label='Prob_cross = 90%')
plot.legend(loc = 'lower right')
plot.title('Fitness vs generación')

#%% Dependencia de la probabilidad de mutacion

C1 = Fitness_calculator(0.5, 0.02, 20)

C2 = Fitness_calculator(0.5, 0.10, 20)

C3 = Fitness_calculator(0.5, 0.5, 20)

#C4 = Fitness_calculator(0.5, 0.80, 20)

plot.figure(2)
plot.plot(C1, label="Prob_mut = 2%")
plot.plot(C2, label='Prob_mut = 10%')
plot.plot(C3, label='Prob_mut = 50%')
#plot.plot(C4, label='Prob_mut = 80%')
plot.legend(loc = 'lower right')
plot.title('Fitness vs generación')

#%% Dependencia a la cantidad de individuos de la población

C1 = Fitness_calculator(0.5, 0.1, 15)

C2 = Fitness_calculator(0.5, 0.1, 20)

C3 = Fitness_calculator(0.5, 0.1, 50)

C4 = Fitness_calculator(0.5, 0.1, 80)

plot.figure(3)
plot.plot(C1, label="Pop_num = 15")
plot.plot(C2, label='Pop_num = 20')
plot.plot(C3, label='Pop_num = 50')
plot.plot(C4, label='Pop_num = 80')
plot.legend(loc = 'lower right')
plot.title('Fitness vs generación')
