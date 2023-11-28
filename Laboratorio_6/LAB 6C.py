# Demostraciones de Crossovers y Mutadores de Pyevolve

from pyevolve import G1DList
from pyevolve import Initializators
from pyevolve import Crossovers
from pyevolve import Mutators
from pyevolve import Selectors
from pyevolve import GSimpleGA

def print_genes(genome):
    ele = ""
    for i in genome[0]:
        ele += " "+str(i)
    print("Elementos del genoma: "+ele)
    
def print_genes_B(genome):
    ele = ""
    for i in genome:
        ele += " "+str(i)
    print("Elementos del genoma: "+ele)

#%% Crossover Binarios
print("----------------------------------------------------------------------")
print("\n")
print("Resultados de cruzamientos Binarios:")
print("\n")

MAbin = G1DList.G1DList(8)
MAbin.setParams(rangemin = 0 , rangemax = 1)
MAbin.genomeList = [0, 0, 0, 0, 0, 0, 0, 0]
PAbin = G1DList.G1DList(8)
PAbin.setParams(rangemin = 0 , rangemax = 1)
PAbin.genomeList = [1, 1, 1, 1, 1, 1, 1, 1]

SON1 = Crossovers.G1DBinaryStringXSinglePoint(None, mom = MAbin, dad = PAbin, count = 1 )
print("Resultado de G1DBinaryStringXSinglePoint")
print_genes(SON1)
print("\n")

SON2 = Crossovers.G1DBinaryStringXTwoPoint(None, mom = MAbin, dad = PAbin, count = 1 )
print("Resultado de G1DBinaryStringXTwoPoint")
print_genes(SON2)
print("\n")

SON3 = Crossovers.G1DBinaryStringXUniform(None, mom = MAbin, dad = PAbin, count = 1 )
print("Resultado de G1DBinaryStringXUniform")
print_genes(SON3)
print("\n")

#%% Mutadores Binarios
print("----------------------------------------------------------------------")
print("\n")
print("Resultados de Mutaciones Binarias:")
print("\n")

Gbin = G1DList.G1DList(8)
Gbin.setParams(rangemin = 0 , rangemax = 1)
Gbin.genomeList = [0, 0, 0, 0, 0, 0, 0, 0]

Mutators.G1DBinaryStringMutatorFlip(Gbin, pmut=1)
print("Resultado de G1DBinaryStringMutatorFlip")
print_genes_B(Gbin)
print("\n")

Gbin.genomeList = [1, 1, 1, 1, 0, 0, 0, 0]

Mutators.G1DBinaryStringMutatorSwap(Gbin, pmut=1)
print("Resultado de G1DBinaryStringMutatorSwap")
print_genes_B(Gbin)
print("\n")

#%% Crossover Enteros
print("----------------------------------------------------------------------")
print("\n")
print("Resultados de cruzamientos Enteros (Listas):")
print("\n")

MAint = G1DList.G1DList(8)
MAint.setParams(rangemin = 0 , rangemax = 20)
MAint.genomeList = [1, 2, 3, 4, 5, 6, 7, 8]
PAint = G1DList.G1DList(8)
PAint.setParams(rangemin = 0 , rangemax = 20)
PAint.genomeList = [9, 10, 11, 12, 13, 14, 15, 16]

SON1 = Crossovers.G1DListCrossoverSinglePoint(None, mom = MAint, dad = PAint, count = 1 )
print("Resultado de G1DListCrossoverSinglePoint")
print_genes(SON1)
print("\n")

SON2 = Crossovers.G1DListCrossoverTwoPoint(None, mom = MAint, dad = PAint, count = 1 )
print("Resultado de G1DListCrossoverTwoPoint")
print_genes(SON2)
print("\n")

SON3 = Crossovers.G1DListCrossoverUniform(None, mom = MAint, dad = PAint, count = 1 )
print("Resultado de G1DListCrossoverUniform")
print_genes(SON3)
print("\n")

#%% Mutadores Listas
print("----------------------------------------------------------------------")
print("\n")
print("Resultados de Mutaciones Para Listas:")
print("\n")

Gint = G1DList.G1DList(8)
Gint.setParams(rangemin = 0 , rangemax = 10)
Gint.genomeList = [0, 1, 2, 3, 4, 5, 6, 7]

Mutators.G1DListMutatorIntegerBinary(Gint, pmut=1)
print("Resultado de G1DListMutatorIntegerBinary")
print_genes_B(Gint)
print("\n")

Gint.genomeList = [1, 2, 3, 4, 5, 6, 7, 8]

Mutators.G1DListMutatorIntegerGaussian(Gint, pmut=1)
print("Resultado de G1DListMutatorIntegerGaussian")
print_genes_B(Gint)
print("\n")

Gint.genomeList = [1, 2, 3, 4, 5, 6, 7, 8]

Mutators.G1DListMutatorIntegerRange(Gint, pmut=1)
print("Resultado de G1DListMutatorIntegerRange")
print_genes_B(Gint)
print("\n")

Gint.genomeList = [1, 2, 3, 4, 5, 6, 7, 8]

Mutators.G1DListMutatorRealGaussian(Gint, pmut=1)
print("Resultado de G1DListMutatorRealGaussia")
print_genes_B(Gint)
print("\n")

Gint.genomeList = [1, 2, 3, 4, 5, 6, 7, 8]

Mutators.G1DListMutatorRealRange(Gint, pmut=1)
print("Resultado de G1DListMutatorRealRange")
print_genes_B(Gint)
print("\n")

Gint.genomeList = [1, 2, 3, 4, 5, 6, 7, 8]

Mutators.G1DListMutatorSwap(Gint, pmut=1)
print("Resultado de G1DListMutatorSwap")
print_genes_B(Gint)
print("\n")


