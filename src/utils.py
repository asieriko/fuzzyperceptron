#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 18:33:36 2020

@author: asier
"""
import csv
import os
import sys
import time
from humanfriendly import format_timespan

import numpy as np


def bintofloat(x):
    """
    float in 0-1 range
    """
    return int("".join([str(y) for y in x]), 2)/(2**len(x)-1)


def individualtofloat(x,precision=10):
    if len(x) % precision != 0:
        raise ValueError("Input should be a multiple of {}".format(precision))
    floats = []
    for i in range(0,len(x),precision):
        floats.append(bintofloat(x[i:i+precision]))
    return floats


def generateGMeasure(mu,l):
    # input: [0.624801, 0.029406, 0.326642, 0.374676]
    # output: [1, 0.9897835708742679, 0.8451411750798067, 0.624801]
    # Computes g(Ej) = 1/lambda * (Product(1+lambda*gi)-1) #i=j..n
    # mu should be sorted as x
    """
    Generates fuzzy densities given :math:`g_{i}=\\mu_{i}`
    
    only needed :math:`g(E_{j})` are calculated, given that the measure is
    sorted to calculate the fuzzy integral
    
    :math:`g(E_{j}) = g_{j} + g(E_{j+1})+\\lambda g_{j}g(E_{j+1})`.
    
    :math:`g(E_{j}) = \\dfrac{1}{\\lambda}*[\\displaystyle\\prod_{j=1}^{n}(1+\\lambda * g_{i}) -1]`."""
    nm = np.array(mu)
    nm[-1] = mu[-1]*l+1
    for i in reversed(range(len(mu)-1)):
        nm[i] = nm[i+1]*(1+l*nm[i])
    # nm = [max(0, min(x, 1)) for x in (nm-1)/l] #Needed or not?
    # nm = np.clip((nm-1)/l, 0, 1) #Needed or not? and slower
    nm = (nm-1)/l
    nm[0] = 1
  
    return nm

def bisectionWang(F,DF,W,epsilon=0.000001):
    # I add epsilon, because it doesn't always get to 0, but e-17
    """
    J. C. Wang, T. Y. Chen and H. M. Shen, Using fuzzy densities
    to determine the λ -value for λ -fuzzy measures, in 9th National
    Conference on Fuzzy Theory and its Applications, 2001, pp. 54–59.
    
    :param F: F(lambda)
    :param DF: F'(lambda)
    :param W: fuzzy measure
    :param epsilon: precision for te approximation to :math:`F(\\lambda)=0`
    
    :rtype: lambda that makes :math:`F(\\lambda)=0` with given measure
    
    
    """
    F1 = round(DF(0,W),10)  # Fprime returns 8e-17 instead of 0
    if F1 == 0:
        return 0
    elif F1 > 0:
        p = -1
        m = 0
    elif F1 < 0:
        p = 1
        m = 0
        while F(p,W) < 0: #FIXME: How exactly?
            # IF F(p) < 0, let m=p,p=p*2 and continue Step 4 (repeat double p until F(0)<0)
            m = p
            p = p * 2
    F2 = F((p + m) / 2,W)
    while abs(F2) > epsilon:
        # print("F2 =",F2,"[p,m]",p,m)
        if F2 > 0:
            p = ( p + m )/ 2
        elif F2 < 0:
            m = ( p + m )/ 2
        F2 = F((p + m) / 2,W)
        # print("F2 ex =",F2,"[p,m]",p,m)
    return (p + m) / 2


def bisectionLien(F,W,epsilon=0.000001):
    """
    Simple algorithm for identifying λ-value for fuzzy measures and fuzzy 
    integrals (Chung-Chang Lien & Chie-Bein Chen)
    
    https://doi.org/10.1080/09720510.2007.10701271
    
    :param F: F(lambda)
    :param W: fuzzy measure
    :param epsilon: precision for te aproximation to F(lambda) = 0
    
    :rtype: lambda that makes F(lambda)=0 with given measure
    """
    F1 = sum(W)
    if F1 == 1:
        return 0
    elif F1 > 1:
        p = -1
        m = 0
    elif F1 < 1:
        p = 1
        m = 0
        while F(p,W) < 0:#antes >= #FIXME: How exactly?
        #IF F(p) < 0, let m=p,p=p*2 and continue Step 4 (repeat double p until F(0)<0)    
            m = p
            p = p * 2
    F2 = F((p + m) / 2,W)
    while abs(F2) > epsilon:
        #print("F2 =",F2,"[p,m]",p,m)
        if F2 > 0:
            p = ( p + m )/ 2
        elif  F2 < 0:
            m = ( p + m )/ 2
        F2 = F((p + m) / 2,W)
        #print("F2 ex =",F2,"[p,m]",p,m)
    return (p + m) / 2


def F(l,mu):
    """:math:`F(\\lambda) = \\displaystyle\\prod_{j=1}^{n}(1+\\lambda * \\mu_{j}) - \\lambda -1`."""
    p = 1
    for m in mu:
        p *= (1+m*l)
    p += -l-1
    return p


def Fprime(l,mu):
    """:math:`F'(\\lambda) = \\sum_{i}(\\mu_{i}\\displaystyle\\prod_{j=1 \\neq i}^{n}(1+\\lambda * \\mu_{j})) - 1`."""
    df = -1
    for i in range(len(mu)):
        p = mu[i]
        for j in range(len(mu)): 
            if i != j:
                p *= (1+mu[j]*l)
        df += p
    return df


class LogStats():
    """
    This class registers stats for the running learning experiment
    It can store several repetitions and for each one serveral data partition
    """

    def __init__(self):
        self.repetition = 0
        self.startTime = {self.repetition: time.time()}
        self.cum_time = {self.repetition: 0}
        self.cum_succ = {self.repetition: []}
        self.LastElapsedTime = {self.repetition: 0}
        self.extra_fields = {self.repetition: []}  # For ouput
        self.extra_headers = []

    def startLogging(self):
        self.startTime[self.repetition] = time.time()

    def setExtraFields(self, headers):
        self.extra_headers = headers

    def update(self, successes, total, fields=None):
        self.LastElapsedTime[self.repetition] = time.time() - self.startTime[self.repetition]
        self.cum_time[self.repetition] += self.LastElapsedTime[self.repetition]
        self.cum_succ[self.repetition].append([successes, total])
        if fields:
            self.extra_fields[self.repetition] = fields
            if len(fields) == len(self.extra_headers):
                print("ERROR: {} fields for {} Headers".format(len(fields), len(self.extra_headers)))

    def printStats(self):
        successes = self.cum_succ[self.repetition][-1][0]
        total = self.cum_succ[self.repetition][-1][1]
        print("Elapsed time last loop: {} - Total elapsed time: {}".format(
            format_timespan(self.LastElapsedTime[self.repetition]),
            format_timespan(self.cum_time[self.repetition])))
        print("{} successes from {} tests, success rate = {:.2f}% Accumulated success rate: {:.2f}% in {} runs".format(
            successes,
            total,
            successes * 100 / total,
            np.average([x / y for x, y in self.cum_succ[self.repetition]]) * 100,
            len(self.cum_succ[self.repetition])))

    def printRepetitionStats(self):
        print("Average success rate for {} runs: {:.2f}% - in {}".format(
            len(self.cum_succ[self.repetition]),
            np.average([x / y for x, y in self.cum_succ[self.repetition]]) * 100,
            format_timespan(self.cum_time[self.repetition])))

    def printFinalStats(self):
        for repetition in range(self.repetition + 1):
            print("Average success rate for {} runs: {:.2f}% - in {}".format(
                len(self.cum_succ[repetition]),
                np.average([x / y for x, y in self.cum_succ[repetition]]) * 100,
                format_timespan(self.cum_time[repetition])))
        print("Average success rate for {} repetitions: {:.2f}% - in {}".format(
            self.repetition + 1,
            np.average([np.average([x / y for x, y in self.cum_succ[i]]) for i in self.cum_succ.keys()]) * 100,
            format_timespan(sum(self.cum_time))))

    def saveStatsToFile(self, file, name):
        # options to create, append, maybe check if it has header
        # write test name, time, final accuray, each one of the partial results
        # How to do for 10 repetitions, maybe a dict o a list o lists
        # Foreach repetiton a new row...
        if not os.path.isfile(file):
            with open(file, 'w', newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerow(["name", "repetition", "Accuracy %", "Time"] + self.extra_headers + ["Accuracies (Folds)"])
        with open(file, 'a', newline='') as myfile:
            wr = csv.writer(myfile)
            for repetition in range[self.repetition + 1]:
                wr.writerow([name, repetition, self.average(repetition), self.elapsed(repetition)] + self.extra_fields[
                    repetition] + np.average([x / y for x, y in self.cum_succ[self.repetition]]) * 100)

    def newRepetition(self):
        self.repetition += 1

    def average(self, rep=None):
        if rep:
            return np.average([x / y for x, y in self.cum_succ[rep]]) * 100
        else:
            return np.average([np.average([x / y for x, y in self.cum_succ[i]]) for i in self.cum_succ.keys()]) * 100

    def elapsed(self, rep=None):
        if rep:
            return self.cum_time[rep]
        else:
            return np.average([np.average([x / y for x, y in self.cum_succ[i]]) for i in self.cum_succ.keys()]) * 100


class Logger(object):
    """
    Log the output from the terminal to a file
    usage: sys.stdout = Logger()
    """
    def __init__(self):
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        with open("logfile.log", "a") as f:
            f.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


