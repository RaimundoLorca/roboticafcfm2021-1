#LAB 7 - Robótica 2021-1 - ALFORITMOS GENETICOS

from pyevolve import G1DList
from pyevolve import GSimpleGA

from pyevolve import Initializators
from pyevolve import Crossovers
from pyevolve import Mutators
from pyevolve import Selectors

import pandas as pd
import numpy as np

from utils import plot_evolution

sentence = '''
'Muy frío, no habíamos visto un frío tanto este año como año tanto frío'
'''
# Unicode
numeric_sentence = list(map(ord, sentence))

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

# Función objetivo
def eval_fitness(genome):
    
    '''
    
      -> float

      Calcula el fitness a partir del grado de diferencia entre el indiviuo
      y la sentence definida.
    
      :param list genome: lista de enteros que contiene el genoma 
      del individuo.
    
      :return: fitness del individuo
    
    '''    
    # Comparamos el individuo con la frase entregada
    score = 0.0
    for a, b in zip(genome, numeric_sentence):
        score += abs(a-b)
    
    # Evitamos que se indetermine (1/score).
    # Notar que para el individuo perfecto, el score es cero
    # en este pto del codigo
    score = score + 1                      
    
    return (1/score)*100

#%% Definir todas las variables para correr el algoritmo

# Número de individuos que contiene la población
pop_num = 60

# Limite inferior y superior que definen el intervalo del dominio
n_min = min(numeric_sentence)
n_max = max(numeric_sentence)

# Probabilidad e cruzamiento y mutación
prob_cross = 0.7
prob_mut = 0.02

# Cantidad de iteraciones a ejecutar
gen_iter = 900

# Largo individuo
genoma_size = len(sentence)

# Inicializar DataFrame que guardará la información de la evolución del 
# algoritmo genético
GA_STATS = pd.DataFrame(index=np.arange(gen_iter), columns=['Max', 'Min', 'Mean'])

#%% Configurar características del algorimto genético

# Configurar genoma (lista 1D de 50 elementos)
genome = G1DList.G1DList(genoma_size)

# Setear el rango max y min de la lista 1D 
genome.setParams(rangemin = n_min, rangemax = n_max,
                bestrawscore=0.00,
                gauss_mu=1, gauss_sigma=4)

# Configurar evaluador del modelo (función eval_fitness)
genome.evaluator.set(eval_fitness)

# Configurar métodos de inicialización, cruzamiento y mutación
genome.initializator.set( Initializators.G1DListInitializatorInteger )
genome.crossover.set( Crossovers.G1DListCrossoverSinglePoint )
genome.mutator.set( Mutators.G1DListMutatorIntegerGaussian )

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

# Ejecutar algoritmo genético
ga_run.stepCallback.set(evolve_callback)
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
print('\nFrase detectada: ', final_sentence)

# Plotear evolución del algoritmo genético y resultado
plot_evolution(GA_STATS)