#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.


#    example which maximizes the sum of a list of integers
#    each of which can be 0 or 1

import random
#1st change import array for individuals
import array


from deap import base
from deap import creator
from deap import tools

# Change 8: Custom fitness functions
import numpy as np
import FIntegrals
from utils import individualtofloat, bisectionLien
from  Datasets import Datasets

ds = Datasets()
x,y = ds.BreastCancerWisconsin()
FI = FIntegrals.ChoquetLambda
atrib = len(x[0])+1

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# 1st change, from individuals fron list to array:
# Speed's up (oneMax 100, from 109secs to 10, accuracy 84.7% - size 1000)    
# creator.create("Individual", list, fitness=creator.FitnessMax)
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator 
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
# toolbox.register("individual", tools.initRepeat, creator.Individual, 
#     toolbox.attr_bool, 1000)

#Change 11: ind size based on n atrib
#10* -> so binary precision is 3 digits
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_bool, 10*atrib)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# the goal ('fitness') function to be maximized
def evalOneMax(individual):
    return sum(individual)*100/len(individual),

#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", evalOneMax)

# register the crossover operator
# Change 2: Frox 2xpoint to 1xpoint
# Accuracy down to 79.7
# toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mate", tools.cxOnePoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
# Change 4: Selecction from Tournament to Roulette
# toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("select", tools.selRoulette)

#----------

# Change 8: Custom fitness functions
# FFISLP Evaluaciion function

def FFISLP(individual):
    wca = 1
    we = 0.1
    floats = individualtofloat(individual)
    weights = floats[:-1]
    cutvalue = floats[-1]
    CA = 0 #number of correctly classified training patterns
    E = 0  #square error between the actual and desired outputs of individual training patterns
    l = bisectionLien(FIntegrals().FLambda, np.array(weights))
    # 3% slower:
    # yout = [1 if FI(xi, weights, l) < cutvalue else 0 for xi in x ]
    # print(sum(yout==y))
    # print(sum((yout-y)**2)**(1/2))

    for xi, yi in zip(x, y):
        CFI = FI(xi, weights, l)
        y_out = 1 if CFI < cutvalue else 0
        if y_out == yi:
        #if ((CFI < cutvalue) and (yi == 1)) or ((CFI >= cutvalue) and (yi == 0)):
            CA += 1#CA(CFISLPik) correctly clasified instances
        E += (y_out-yi)**2
        # E += (CFI-yi)**2

    E = E**(1/2)

    return wca * CA - we * E, #, because in fitness assignemt it asked for same as weihts

toolbox.register("evaluate", FFISLP)


#----------

def main():
    random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=300)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    #Change 3:Mutation from 0.2->1
    # So all generated offspring is candidate for mutations
    MUTPB = 0.2
    # Change 5: CxPB from 0.5 to 0.95
    CXPB = 0.95
    #CXPB, MUTPB = 0.5, 0.2
    
    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    print(fitnesses)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    
    # Begin the evolution
    #Change 9 -> max 500 generations
    # while max(fits) < 100 and g < 1000:
    while g < 500:
        # Change 6: Elitism
        Ndel = 2
        elite = list(map(toolbox.clone, tools.selBest(pop,Ndel)))

        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        #Change 7: Mutation for new children onlly
        
                if random.random() <= MUTPB:
                    toolbox.mutate(child1)
                if random.random() <= MUTPB:
                    toolbox.mutate(child2)
        # for mutant in offspring:

        #     # mutate an individual with probability MUTPB
        #     if random.random() < MUTPB:
        #         toolbox.mutate(mutant)
        #         del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
                
        #Change 6: Elitims (part2)
        for _ in range(Ndel):
            pop.remove(random.choice(pop))
        pop = pop + elite


        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    main()
