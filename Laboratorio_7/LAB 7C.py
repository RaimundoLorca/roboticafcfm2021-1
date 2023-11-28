#LAB 7 - Robótica 2021-1 - ALFORITMOS GENETICOS

from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import Initializators
from pyevolve import Crossovers
from pyevolve import Mutators
from pyevolve import Selectors

import random
import pandas as pd
import numpy as np

from utils.chess import ChessBoard
from utils import plot_evolution

#%% Funciones extras

# Step callback function, sirve para guardar el progeso del AG.
# Se llama al final de cada iteración (generación) del algoritmo
def evolve_callback(ga_engine):
    
    # Generación actual
    generation = ga_engine.getCurrentGeneration()
    
    # Extraer estadísticas
    gen_stats = ga_run.getStatistics()
   
    global GA_STATS

    GA_STATS.at[generation, 'Max'] = gen_stats['rawMax']
    GA_STATS.at[generation, 'Min'] = gen_stats['rawMin']
    GA_STATS.at[generation, 'Mean'] = gen_stats['rawAve']
   
    return False

# Función para visualizar el tablero de ajedrez
def draw_chessboard(best):
    
    # Incializar tablero
    board = ChessBoard()
    
    for i, j in enumerate(best):
        # Agregar al tablero
        board.drawPiece('black_queen', position=(j, i))
        
    # Visualizar tablero
    board.plot()
    
#%% Planteamiento del problema

# The "n" in n-queens (pueden probar con un tablero mas grande, pero 
# tienen que comentar la ultima linea del codigo, ya que el visualizador 
# solo está implementado para tableros 8x8)
BOARD_SIZE = 8

# Inicializar de orden
def Queens_init(genome, **args):
    
    """
      -> pyevolve.genome
    
      Genera un orden aleatorio de la lista de valores [0, 1, ..., n-1] donde n
      corresponde al tamaño del tablero (nxn). 
    
      :param pyevolve.genome genome: genoma G1DList a inicializar.
    
      :return: genoma modificado.
      
    """
    
    # Generar lista del tipo [0, 1, ..., n-1] y reordenarla aleatoriamente
    shuffle_list = list(range(BOARD_SIZE))
    random.shuffle(shuffle_list)
    
    genome.genomeList = shuffle_list

    return genome
       

# The n-queens fitness function
def eval_fitness(genome):
    
    """
      -> pyevolve.genome
    
      Determina la cantidad de reinas que no están siendo amenazas 
      por otra de las piezas.
    
      :param pyevolve.genome genome: individuo al cual se le calcula el fitness.
    
      :return: int, número de piezas a salvo.
      
    """
      
    collisions = 0
    # Restringir que cada reina esté en una fila distinta
    for i in range(BOARD_SIZE):
       if i not in genome: return 0
       
    # Determinar si hay más de una reina en una diagonal
    for i in range(BOARD_SIZE):
       col = False
       for j in range(BOARD_SIZE):
          if (i != j) and (abs(i-j) == abs(genome[j]-genome[i])):
             col = True
       if col == True: collisions +=1
       
    return BOARD_SIZE-collisions


#%% Definir todas las variables para correr el algoritmo

# Número de individuos que contiene la población
pop_num = 15

# Probabilidad e cruzamiento y mutación
prob_cross = 0.8
prob_mut = 0.05

# Cantidad de iteraciones a ejecutar
gen_iter = 250

# Inicializar DataFrame que guardará la información de la evolución del 
# algoritmo genético
GA_STATS = pd.DataFrame(index=np.arange(gen_iter), columns=['Max', 'Min', 'Mean'])

#%% Configurar características del algorimto genético

# Configurar genoma (lista 1D de 50 elementos)
genome = G1DList.G1DList(BOARD_SIZE)

# Setear el rango max y min de la lista 1D 
genome.setParams(bestrawscore=BOARD_SIZE, rounddecimal=2)

# Configurar evaluador del modelo (función eval_fitness)
genome.evaluator.set(eval_fitness)

# Configurar métodos de inicialización, cruzamiento y mutación
genome.initializator.set( Queens_init )
genome.crossover.set( Crossovers.G1DListCrossoverEdge )
genome.mutator.set( Mutators.G1DListMutatorSwap )

#Crossovers.G1DListCrossoverEdge

# Configurar ejecución del algoritmo genético
ga_run = GSimpleGA.GSimpleGA(genome)
ga_run.setGenerations(gen_iter)

# Configurar población
ga_run.setPopulationSize(pop_num)

# Configurar probabilidades
ga_run.setCrossoverRate(prob_cross)
ga_run.setMutationRate(prob_mut)

# Configurar método de selección
ga_run.selector.set( Selectors.GRankSelector )

# Configurar el criterio de termino
ga_run.terminationCriteria.set(GSimpleGA.RawScoreCriteria)
ga_run.stepCallback.set(evolve_callback)

# Ejecutar algoritmo genético
ga_run.evolve()

# Obtener mejor individuo
bestIndividual = ga_run.bestIndividual()
final_sentence = str().join(map(chr, bestIndividual))

BEST_FITNESS = bestIndividual.getFitnessScore()
BEST_SCORE = bestIndividual.getRawScore()
BEST_GENOME = bestIndividual.getInternalList()

# Generación de convergencia
gen_convergencia = ga_run.getCurrentGeneration()

# Display resultados
print("----------------------------------------------------------------------")
print("Resultados del algoritmo genético:")
print('\nGeneration: {:04d}'.format(gen_convergencia))
print('\nBest genome: ', BEST_GENOME)
print('\nScore: {:.3f}'.format(BEST_SCORE))
print('\nFitness: {:.3f}'.format(BEST_FITNESS))
print(final_sentence)

# Plotear evolución del algoritmo genético y resultado
plot_evolution(GA_STATS)

# Plotear tablero de ajedrez 
draw_chessboard(bestIndividual)


