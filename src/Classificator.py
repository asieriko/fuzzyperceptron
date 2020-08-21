#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:40:38 2020

@author: asier
"""
import time
from humanfriendly import format_timespan


from sklearn.model_selection import KFold, train_test_split
import numpy as np

from Datasets import Datasets
import pFISLP as FISLP
import utils
from FIntegrals import FIntegrals

class Classificator():

    def __init__(self, FI):
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

    def test(self, individual, x_test, y_test):
        floats = utils.individualtofloat(individual)
        weights = floats[:-1]
        cutvalue = floats[-1]
        FI = FIntegrals()
        l = utils.bisectionLien(FI.FLambda, np.array(weights))
        successes = 0
        for x_testi, y_testi in zip(x_test, y_test):
            CFI = self.FI(x_testi, weights, l) #With landa less expensive testing...
            y_out = 1 if CFI < cutvalue else 0
            if y_out == y_testi:
                successes += 1
        return successes, len(x_test)

    def Folds(self, x, y, splits=10):
        kf = KFold(n_splits=splits)
        cum_time = 0
        cum_succ = []
        for train_index, test_index in kf.split(x):
            start = time.time()
            successes = 0
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            iFISLP = FISLP.FISLP(self.NPOP, self.NCON, self.Ndel, self.IND_SIZE,
                                 self.Prc, self.Prm, self.wca, self.we, x_train, y_train, self.FI)
            best, pop, log = iFISLP.GAFISLP()
            successes, total = self.test(best[0], x_test, y_test)

            elapsed = time.time() - start
            cum_time += elapsed
            cum_succ.append([successes, total])
            print("Elapsed time last loop:", format_timespan(elapsed),
                  "Total elapsed time:", format_timespan(cum_time))
            print(successes, "successes from ", len(x_test), " success rate =",
                  successes*100/len(x_test), "% Accumulated success rate:",
                  np.average([x/y for x, y in cum_succ])*100, "% in",len(cum_succ),"uns")
            
        print("Average success rate for", splits, "runs:",
              np.average([x/y for x, y in cum_succ])*100, "% - in", format_timespan(cum_time))


    def Divide(self, x, y, size=5, times=10):
        cum_time = 0
        cum_succ = []
        for i in range(times):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size)
            start = time.time()
            successes = 0
            iFISLP = FISLP.FISLP(self.NPOP, self.NCON, self.Ndel, self.IND_SIZE,
                                 self.Prc, self.Prm, self.wca, self.we, x_train, y_train, self.FI)
            best, pop, log = iFISLP.GAFISLP()
            successes, total = self.test(best[0], x_test, y_test)

            elapsed = time.time() - start
            cum_time += elapsed
            cum_succ.append([successes, total])
            print("Elapsed time last loop:", format_timespan(elapsed),
                  "Total elapsed time:", format_timespan(cum_time))
            print(successes, "successes from ", len(x_test), " success rate =",
                  successes*100/len(x_test), "% Accumulated success rate:",
                  np.average([x/y for x, y in cum_succ])*100, "% in",len(cum_succ),"uns")
            
        print("Average success rate for", times, "runs:",
              np.average([x/y for x, y in cum_succ])*100, "% - in", format_timespan(cum_time))


#https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
import sys

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        

    def write(self, message):
        self.terminal.write(message)
        with open("logfile.log", "a") as f:
            f.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

import datetime

if __name__ == "__main__":
    sys.stdout = Logger()
    FI = FIntegrals()
    F = FI.ChoquetLambda
    classi = Classificator(F)
    # print("---------------- Choquet ------------- ")
    # print(datetime.datetime.now())
    # print("---------------- Cleveland HD ------------- ")
    # x, y = classi.ds.ProcessedClevelandHD()
    # classi.Folds(x, y)
    # print(datetime.datetime.now())
    # print("---------------- Statlog Heart ------------ ")
    # x, y = classi.ds.StatlogHeart()
    # classi.Folds(x, y)
    # print(datetime.datetime.now())
    # print("----------------- Breast Cancer------------ ")
    # x, y = classi.ds.BreastCancer()
    # # classi.Folds(x, y)
    # print("----------------- 50% ------------ ")
    # classi.Divide(x, y)
    # print(datetime.datetime.now())
    # print("----------------- 10-Folds------------ ")
    # classi.Folds(x, y)
    # print(datetime.datetime.now())
    # print("---------------- Appendicitis ------------- ")
    # x, y = classi.ds.Appendicitis()
    # classi.Folds(x, y, splits=len(y))
    # print(datetime.datetime.now())
    # print("---------------- Diabetes ----------------- ")
    # x, y = classi.ds.Diabetes()
    # classi.Folds(x, y)
    # print(datetime.datetime.now())
    # print("---------------- Breast Wisconsin --------- ")
    # x, y = classi.ds.BreastCancerWisconsin()
    # classi.Folds(x, y)
    # print(datetime.datetime.now())
    print("----------------- Thyroid ----------------- ")
    x, y = classi.ds.Thyroid()
    classi.Folds(x, y)
    print(datetime.datetime.now())
    print("---------------- German Credit ------------ ")
    x, y = classi.ds.GermanCredit()
    classi.Folds(x, y)
    print(datetime.datetime.now())
    
    print(datetime.datetime.now())
    print("---------------- Sugeno ------------- ")
    F = FI.SugenoLambda
    classi = Classificator(F)
    print(datetime.datetime.now())
    print("---------------- Cleveland HD ------------- ")
    x, y = classi.ds.ProcessedClevelandHD()
    classi.Folds(x, y)
    print(datetime.datetime.now())
    print("---------------- Statlog Heart ------------ ")
    x, y = classi.ds.StatlogHeart()
    classi.Folds(x, y)
    print(datetime.datetime.now())
    print("----------------- Breast Cancer------------ ")
    x, y = classi.ds.BreastCancer()
    # classi.Folds(x, y)
    print("----------------- 50% ------------ ")
    classi.Divide(x, y)
    print(datetime.datetime.now())
    print("----------------- 10-Folds------------ ")
    classi.Folds(x, y)
    print("---------------- Appendicitis ------------- ")
    x, y = classi.ds.Appendicitis()
    classi.Folds(x, y, splits=len(y))
    print(datetime.datetime.now())
    print("---------------- Diabetes ----------------- ")
    x, y = classi.ds.Diabetes()
    classi.Folds(x, y)
    print(datetime.datetime.now())
    print("---------------- Breast Wisconsin --------- ")
    x, y = classi.ds.BreastCancerWisconsin()
    classi.Folds(x, y)
    print(datetime.datetime.now())
    print("----------------- Thyroid ----------------- ")
    x, y = classi.ds.Thyroid()
    classi.Folds(x, y)
    print(datetime.datetime.now())
    print("---------------- German Credit ------------ ")
    x, y = classi.ds.GermanCredit()
    classi.Folds(x, y)
    print(datetime.datetime.now())
