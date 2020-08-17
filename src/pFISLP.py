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

# Parallelized version

import array
import random

import numpy as np

from operator import attrgetter
from concurrent.futures import  ProcessPoolExecutor

from deap import base
from deap import creator
from deap import tools

from FIntegrals import FIntegrals
from utils import individualtofloat,bisectionLien

class FISLP():
    
    def __init__(self,NPOP,NCON,Ndel,IND_SIZE,Prc,Prm,wca,we,x,y,FI):
        #x,y = ds.Appendicitis()
        #or dataset?
        #x,y = ds.BreastCancerWisconsin()
        #Algorithm parameters
        #5.2. Pre-specified parameter specifications
        self.NPOP = NPOP  #Population size
        self.NCON = NCON #Total number of generations
        self.Ndel = Ndel# for not generating much perturbation in the next generation, 
        #a small number of the elite chromosomes is taken into account.
        self.IND_SIZE = IND_SIZE #for 3 decimal precision in binary string
        self.Prc = Prc
        self.Prm = Prm
        #Evaluation function parameters
        self.wca = wca
        self.we = we
        #CFISLP is handled as an individual consisting of (n + 1) substrings
        self.NATTR = len(x[0])+1 # +1 for the cut parameter
        
        #Dataset
        self.x = x
        self.y = y
        
        #Fuzzy Integral F(x,mu,lambda)
        self.FI = FI

        self.configureDEAP()

    def FFISLP(self,individual):
        floats = individualtofloat(individual)
        weights = floats[:-1]
        cutvalue = floats[-1]
        CA = 0 #number of correctly classified training patterns
        E = 0  #square error between the actual and desired outputs of individual training patterns       
        l = bisectionLien(FIntegrals().FLambda,np.array(weights))
        for xi,yi in zip(self.x,self.y):
            CFI = self.FI(xi,weights,l)
            y_out = 1 if CFI < cutvalue else 0
            if y_out == yi:
            #if ((CFI < cutvalue) and (yi == 1)) or ((CFI >= cutvalue) and (yi == 0)):
                CA += 1#CA(CFISLPik) correctly clasified instances
            E += (y_out-yi)**2
            # E += (CFI-yi)**2
    
        E = E**(1/2)
        return self.wca * CA - self.we * E

    def fitnessArt(self,individual):
        #return sum(individual), #For fast testing the algorithm
        return self.FFISLP(individual),



    def selRoulette(self,individuals, k, fit_attr="fitness"):
        """Select *k* individuals from the input *individuals* using *k*
        spins of a roulette. The selection is made by looking only at the first
        objective of each individual. The list returned contains references to
        the input *individuals*.
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list of selected individuals.
        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.
        .. warning::
           The roulette selection by definition cannot be used for minimization
           or when the fitness can be smaller or equal to 0.
        """
    
        s_inds = sorted(individuals, key=attrgetter(fit_attr), reverse=True)
        sum_fits = sum(getattr(ind, fit_attr).values[0] for ind in individuals)
        min_fit = getattr(s_inds[0],fit_attr)
        divider = sum_fits - len(s_inds) * min_fit
        probs = [(getattr(x,fit_attr) - min_fit)/divider for x in s_inds]
        chosen = []
        for i in range(k):
            u = random.random() * sum_fits
            sum_ = 0
            for j,ind in enumerate(s_inds):
                sum_ += probs[j]
                if sum_ > u:
                    chosen.append(ind)
                    break
    
        return chosen

    def mateMutateEvaluate(self,child1,child2):
        if random.random() < self.Prc:
            self.toolbox.mate(child1, child2)
        #the mutation operation with a pre-specified probability,
        #Prm, is performed on each bit or gene of the string.
        self.toolbox.mutate(child1)
        self.toolbox.mutate(child2)
        del child1.fitness.values
        del child2.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [child1,child2]
        fitnesses = map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        

    def configureDEAP(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", array.array, typecode='b', 
                       fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        # Attribute generator
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        # Structure initializers
        self.toolbox.register("individual", tools.initRepeat, 
                              creator.Individual, self.toolbox.attr_bool, 
                              self.IND_SIZE * self.NATTR)
        #Each gene in the chromosome is randomly assigned as either 1 or 0, 
        #with the probability of 0.5.
        self.toolbox.register("population", tools.initRepeat, list, 
                              self.toolbox.individual)    
        self.toolbox.register("evaluate", self.fitnessArt)
        self.toolbox.register("mate", tools.cxOnePoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=self.Prm)
        self.toolbox.register("select", tools.selRoulette)#k number o to select
        #toolbox.register("select", tools.selBest)
        return self.toolbox

    def GAFISLP(self):
        #random.seed(64)
        errors=0
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max) 
    
        #An initial population containing __Ncon__? binary chromosomes 
        #is generated and inserted into P0. 
        #Step 2. enerate the initial population of Npop chromosomes.
        pop = self.toolbox.population(n=self.NPOP)
    
        # Step 3. Compute the fitness value of each CFISLP in the current population.
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
    
        #the best chromosome with maximum fitness value over all generations is 
        #taken as the desired solution
        halloffame = tools.HallOfFame(1)
        halloffame.update(pop)
    
    
            # Append the current generation statistics to the logbook
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals','max']
        record = stats.compile(pop)
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
    
    
        for g in range(self.NCON):
            #The total number of generations (i.e., N con) is used as
            #the stopping condition
            
            
            #Step 4. Generate new Npop chromosomes using the genetic operations 
            #for the next population.
    
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
    
            # Parallelized
            with ProcessPoolExecutor(max_workers=6) as executor:
                executor.map(self.mateMutateEvaluate, offspring[::2], offspring[1::2])  
    
            # Not parallelized
            # for child1, child2 in zip(offspring[::2], offspring[1::2]):
            #     self.mateMutateEvaluate(child1,child2)

            #Step 5. Perform an elitist strategy.        
            # The population is  replaced by the offspring,Select Ndel best parents, delete  Ndel random offspring
            #In practice, Ndel (0 6 Ndel 6 Npop) chromosomes are randomly selected,
            #and are removed from the current population (i.e., Pi+1). Subsequently,
            #Ndel chromosomes with the maximum fitness value in the previous population
            #(i.e., Pi) are inserted into the current one
            elite = tools.selBest(pop,self.Ndel)
            pop[:] = offspring
            for _ in range(self.Ndel):
                try:
                    pop.remove(random.choice(pop))
                except:
                    errors+=1
                    # print("offspring")
                    # print(offspring)
                    # print("pop")
                    # print(pop)
            pop += elite
            
            # Compute stats
            halloffame.update(offspring)
            record = stats.compile(pop)
            logbook.record(gen=g+1, nevals=len(invalid_ind), **record)
            #print(logbook.stream)
            
        if errors != 0:
            print("Errors",errors)
        #Step 6. Terminate the algorithm when Ncon generations have been generated;
        #otherwise, return to Step 3.
        return halloffame,pop,logbook