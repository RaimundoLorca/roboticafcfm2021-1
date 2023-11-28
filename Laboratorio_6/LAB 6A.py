#LAB 6 - Robótica 2021-1 - ALFORITMOS GENETICOS

#Cargar librerias a usar
import random 
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import cm

# Función objetivo:
def funcion_objetivo(x,y):
    
    '''
    
        Función objetivo del problema.
        
        :param float x: posición del eje x del punto a evaluar.
    	:param float y: posición del eje y del punto a evaluar..
        
        :return: función objetivo evaluada (float).
    
    '''
    
    r = np.sqrt((x**2)+(y**2)) 
    z = (1/(1+(r**0.7)))*np.cos(y/2.5)*np.sin(x/2.5)
       
    return z + 0.3
    

#%% Graficar función objetivo

x = np.linspace(-20,20,40)
y = np.linspace(-20,20,40)  

X, Y = np.meshgrid(x,y)

Z = funcion_objetivo(X,Y)


norm = plot.Normalize(Z.min(), Z.max())
colors = cm.viridis(norm(Z))
rcount , ccount , _ = colors.shape
    
fig = plot.figure(1)
ax = fig.gca(projection='3d')    
surf = ax.plot_surface(X,Y,Z, rcount=rcount, ccount=ccount, facecolors=colors,
                       shade=False, zorder=0)  
surf.set_facecolor((0,0,0,0))

#%% Definición de la funciónes necesarias para el algoritmo

# Inicializador (Ojo, solo crea un individuo)
def Init(n_min,n_max):
    ''' 
        Función que se encarga de inicializar los individuos que se utilizarán 
        en la población inicial.
        
        Notar que en este caso un individuo consiste en una tupla del tipo
        [pos_x, pos_y], en donde cada valor corresponde a la posición del punto
        en el eje x y en el eje y respectivamente. Ambos valores son floats
        aleatorios entre n_min y n_max.
        
        :param float n_min: límite inferior del rango.
    	:param float n_max: límite superior del rango.
    
    	:return: np.array del tipo [pos_x, pos_y].
    '''
    
    indivX = random.uniform(n_min, n_max)
    indivY = random.uniform(n_min, n_max)
    
    return np.array([indivX, indivY])


# Crossover
def Cross(indiv_1, indiv_2, prob = 0.5):
    
    '''
    
        Función encarcarga del cruzamiento de 2 individuos cualquiera,
        retornando un individuo nuevo con genes de ambos padres.
        
        Para este caso, el cruzamiento consiste en calcular el promedio
        aritmético para cada coordenada entre ambos individuos. Notar que
        se incluye una probabilidad de cruzamiento, si no se cumple la condicion
        para llevar a cabo el cruzamiento, entonces se elige aleatoriamente
        a uno de los padres como el nuevo individuo.
        
        :param np.array indiv_1: padre 1
    	:param np.array indiv_2: padre 2
        .param float prob: probabilidad de cruzamiento (por defecto en 0.5)
    
    	:return: individuo nuevo (np.array del tipo [pos_x, pos_y]).
    
    '''
    
    padres = [indiv_1, indiv_2]
    
    if random.random() < prob:
        new_indiv = (indiv_1 + indiv_2)/2  
        
    else:
        new_indiv = random.choice(padres)
        
    return new_indiv

# Mutador      
def Mutator(indiv, n_min, n_max, prob = 0.1):
    
    '''
    
        Función encargada de la mutación de un individuo, la cual toma un
        individuo y lo varía levemente.
        
        Para este caso, la mutación consiste en retornar un individuo 
        completamente nuevo, lo cual se logra utilizando la función Init.
        Notar que se incluye una probabilidad de mutación, si no se cumple 
        la condicion para llevar a cabo la mutación, entonces el individuo
        no se modifica. 
        
        :param np.array indiv: individuo a mutar
        :param float n_min: límite inferior del rango.
    	:param float n_max: límite superior del rango.
        .param float prob: probabilidad de cruzamiento (por defecto en 0.1)
        
        :return: individuo nuevo (np.array del tipo [pos_x, pos_y]).
    
    '''  
    
    if random.random() < prob:
        new_indiv = Init(n_min,n_max)
        
    else:
        new_indiv = indiv
        
    return new_indiv

# Evaluación
def Eval_fitness(indiv):
    
    '''
        
        Función encargada de evaluar que tan bueno es nuestro individuo, 
        asignando un valor numérico a este (score). Mientras mayor sea el 
        fitness, mejor es el individuo.
        
        Para este caso, el fitness consiste en evaluar la funcion objetivo
        definida anteriormente. 
        
        :param np.array indiv: individuo a evaluar
        
        :return: fitness del individuo evaluado.

    '''
    
    score = funcion_objetivo(indiv[0],indiv[1])
    
    return score

def Selector(population, k=15):
    
    '''
    
        Función encargada de elegir los individuos que se van a reproducir
        para generar la siguiente generación.
        
        Para este caso se utilizará el selector por torneo, el cual toma k 
        individuos de la población incial, calcula su fitness y entrega al 
        individuo mejor evaluado y una lista con todos los otros (k-1) 
        individuos.
        
        :param list(np.array) population: lista con los individuos candidatos
                                          a seleccionar.
        
        :return: tupla del tipo (best_indiv , others)
    
    '''
    
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

def Best(population):
    
    '''
    
        Función encargada de entregar la información del mejor individuo 
        de una población. En específico, el individuo en sí y su fitness.
        
        :param list(np.array) population: lista con los individuos a evaluar.
        
        :return: tupla del tipo (best_fittnes, best_indiv))
    
    '''
    
    scores = []
    
    for indiv in population:
        
        score_indiv = Eval_fitness(indiv)
        scores.append(score_indiv)
         
    index_max = np.argmax(scores)
        
    best_fittnes = scores[index_max]
    best_indiv = population[index_max]
    
    return best_fittnes, best_indiv
    
#%% Definir todas las variables para correr el algoritmo

# Número de individuos que contiene la población
pop_num = 20

# Limite inferior y superior que definen el intervalo del dominio
n_min = -20
n_max = 20

# Probabilidad e cruzamiento y mutación
prob_cross = 0.5
prob_mut = 0.1

# Cantidad de iteraciones a ejecutar
gen_iter = 200

# Inicializar vectores que almacenan la ruta mejores individuos
RutaX = []
RutaY = []
RutaZ = []

#%% Armar la estructura del algoritmo

# Inicializar la población inicial
population = [Init(n_min, n_max) for i in range(pop_num)]

# population =[]

# for i in range(pop_num):
#     new_indiv = Init(n_min, n_max)
#     population.append(new_indiv)

for i in range(gen_iter):
    
    new_population = []
    
    while len(new_population) < pop_num:
        
        # Selección
        best_indiv, rest_tournament = Selector(population)
        
        # Redifinir la siguiente población
        for indiv in rest_tournament:
            
            new_indiv = Cross(best_indiv, indiv, prob = prob_cross)
            
            new_indiv = Mutator(new_indiv, n_min, n_max, prob=prob_mut)
            
            new_population.append(new_indiv)
            
            if len(new_population) == pop_num:
                break
    
    population = new_population

    # Almacenar los datos del mejor individuo de la generación
    best_fitness, best_indiv = Best(population)

    RutaX.append(best_indiv[0])
    RutaY.append(best_indiv[1])
    RutaZ.append(best_fitness)
    
#%% Display de los resultados

# Extraer el mejor individuo
BEST_FITNESS, BEST_INDIV = Best(population)

# Imprimir detalles
print('Generation:', gen_iter)
print('Best genome:' + str(BEST_INDIV))
print('fitness:', BEST_FITNESS)      

# Plotear la ruta de los mejores individuos
best = ax.scatter(BEST_INDIV[0], BEST_INDIV[1], BEST_FITNESS, zorder=10)
ruta = ax.plot3D(RutaX, RutaY ,RutaZ, 'k', linewidth=2, zorder=5)   


# Fin :)

