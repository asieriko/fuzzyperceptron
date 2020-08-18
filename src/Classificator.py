#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:40:38 2020

@author: asier
"""
import time
from humanfriendly import format_timespan


from sklearn.model_selection import KFold
import numpy as np

from Datasets import Datasets
import pFISLP as FISLP
import utils
from FIntegrals import FIntegrals

class Classificator():
    
    def __init__(self,FI):
        self.ds = Datasets()
        
        #Algorithm parameters
        #5.2. Pre-specified parameter specifications
        self.NPOP = 50  #Population size
        self.NCON = 500 #Total number of generations
        self.Ndel = 2# for not generating much perturbation in the next generation, 
        #a small number of the elite chromosomes is taken into account.
        self.IND_SIZE = 10#for 3 decimal precision in binary string
        self.Prc = 0.95#0.5 deap example. works better
        self.Prm = 0.05#0.2 deap example. works better
        #Evaluation function parameters
        self.wca = 1.0
        self.we = 0.1
        #CFISLP is handled as an individual consisting of (n + 1) substrings
        #NATTR = len(x[0])+1 # +1 for the cut parameter
        
        self.FI = FI
    
    def test(self,individual,X_test,y_test):
        floats = utils.individualtofloat(individual)
        weights = floats[:-1]
        cutvalue = floats[-1]
        FI = FIntegrals()
        l = utils.bisectionLien(FI.FLambda, np.array(weights))
        successes = 0
        for x_testi,y_testi in zip(X_test,y_test):
            CFI = self.FI(x_testi,weights,l) #With landa less expensive testing...
            y_out = 1 if CFI < cutvalue else 0
            if y_out == y_testi:
                successes += 1
        return successes,len(X_test)
        
    
    def Folds(self,x,y,splits=10):
        kf = KFold(n_splits=splits)
        cum_time = 0
        cum_succ = []
        for train_index, test_index in kf.split(x):
            start = time.time()
            successes = 0
            X_train, X_test = x[train_index], x[test_index] 
            y_train, y_test = y[train_index], y[test_index]
            iFISLP = FISLP.FISLP(self.NPOP, self.NCON, self.Ndel, self.IND_SIZE, 
                                 self.Prc, self.Prm, self.wca, self.we, X_train, y_train,self.FI)
            best,pop, log = iFISLP.GAFISLP()
            successes,total = self.test(best[0],X_test,y_test)
            # floats = utils.individualtofloat(best[0])
            # weights = floats[:-1]
            # cutvalue = floats[-1]
            # FI = FIntegrals()
            # for x_testi,y_testi in zip(X_test,y_test):
            #     CFI = FI.ChoquetLambda(x_testi,weights)
            #     if ((CFI < cutvalue) and (y_testi == 1)) or ((CFI >= cutvalue) and (y_testi == 0)):
            #         print("Success")
            #         successes += 1
            #     else:
            #         print("Error")
            elapsed = time.time() - start
            cum_time += elapsed
            cum_succ.append([successes,total])
            print("Elapsed time last loop:",format_timespan(elapsed),
                  "Total elapsed time:",format_timespan(cum_time))
            print(successes,"successes from ",len(X_test)," success rate =",
                  successes*100/len(X_test),"% Accumulated success rate:",
                  np.average([x/y for x,y in cum_succ])*100,"%")
        
        print("Average success rate for",splits,"runs:",
              np.average([x/y for x,y in cum_succ])*100,"% - in",format_timespan(cum_time))


    def Appendicitis(self,x,y):       
        successes = 0
        cum_time = 0
        cum_succ = []
        for i in range(len(x)):
            print("{}/{}".format(i+1,len(x)))
            start = time.time()
            X_test = x[i]
            y_test = y[i]
            X_train = x.copy()
            y_train = y.copy()
            X_train = np.delete(X_train,i,axis=0)
            y_train = np.delete(y_train,i)
            iFISLP =FISLP.FISLP(self.NPOP, self.NCON, self.Ndel, self.IND_SIZE, self.Prc, self.Prm, self.wca, self.we, X_train, y_train,self.FI)
            
            best,pop, log = iFISLP.GAFISLP()
            
            #test trained data
            success,total = self.test(best[0],[X_test],[y_test])
            
            # floats = utils.individualtofloat(best[0])
            # weights = floats[:-1]
            # cutvalue = floats[-1]
            # FI = FIntegrals()
            # CFI = FI.ChoquetLambda(X_test,weights)
            # if y_test == 1:
            #     print(CFI,"<",cutvalue,"?")
            # else:
            #     print(CFI,">=",cutvalue,"?")
            # if ((CFI < cutvalue) and (y_test == 1)) or ((CFI >= cutvalue) and (y_test == 0)):
            #     print("Success")
            #     successes += 1
            # else:
            #     print("Error")
            successes += success
            print("Partial",successes,"successes from ",i+1," success rate =",successes*100/((i+1)),"%")
            elapsed = time.time() - start
            cum_time += elapsed
            cum_succ.append([success,total])
            print("Elapsed time last loop:",format_timespan(elapsed),"Total elapsed time:",format_timespan(cum_time))
            
        print(successes,"successes from ",len(x)," success rate =",successes*100/len(x),"%")
        print("Average success rate for",len(x),"runs:",np.average([x/y for x,y in cum_succ])*100,"% - in",format_timespan(cum_time))
            
if __name__ == "__main__":
    FI = FIntegrals()
    F = FI.SugenoLambda
    classi = Classificator(F)
    # print("----------------- Thyroid ----------------- ")
    # x,y = classi.ds.Thyroid()
    # classi.Folds(x,y)
    # print("----------------- Breast Cancer------------ ")
    # x,y = classi.ds.BreastCancer()
    # classi.Folds(x,y)
    # print("---------------- Appendicitis ------------- ")
    x,y = classi.ds.Appendicitis()
    classi.Appendicitis(x,y)
    # print("---------------- Statlog Heart ------------ ")
    # x,y = classi.ds.StatlogHeart()
    # classi.Folds(x,y)
    # print("---------------- Appendicitis ------------- ")
    # x,y = classi.ds.Appendicitis()
    # classi.Folds(x,y,splits=len(y))
    # print("---------------- Diabetes ----------------- ")
    # x,y = classi.ds.Diabetes()
    # classi.Folds(x,y)
    #print("---------------- Cleveland HD ------------- ")
    #x,y = classi.ds.ProcessedClevelandHD()
    #classi.Folds(x,y)
    #print("---------------- Breast Wisconsin --------- ")
    #x,y = classi.ds.BreastCancerWisconsin()
    #classi.Folds(x,y)
    #print("---------------- German Credit ------------ ")
    #x,y = classi.ds.GermanCredit()
    #classi.Folds(x,y)        
