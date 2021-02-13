#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:40:38 2020

@author: asier
"""
import time
from humanfriendly import format_timespan

import torch

from sklearn.model_selection import KFold, train_test_split
import numpy as np

from Datasets import Datasets
import pFISLP as FISLP
import utils
from FIntegrals import FIntegrals
import Choquet_integral_nn_torch as CIT


class CITClassificator():
    """
    This classificator uses a NN with a Choquet Integral as neuron
    It estimates all parameters of the fuzzy measure, alone and combined
    But it requieres a lot of memory
    """

    def __init__(self):
        self.ds = Datasets()

    def test(self, net, x_test, y_test):
        successes = 0
        for x, y in zip(x_test, y_test):
            y_pred = net.forward(torch.tensor([x], dtype=torch.float))

            if ((y == 1) and (y_pred >= 0.5)) or ((y == 0) and (y_pred < 0.5)):
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

            N_in = len(x_train[0])
            N_out = 1

            net = CIT.Choquet_integral(N_in, N_out)

            # set the optimization algorithms and paramters the learning
            learning_rate = 0.3

            # Construct our loss function and an Optimizer. The call to model.parameters()
            # in the SGD constructor will contain the learnable parameters of the two
            # nn.Linear modules which are members of the model.
            criterion = torch.nn.MSELoss(reduction='mean')
            optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

            num_epochs = 600

            # convert from numpy to torch tensor
            X_train = torch.tensor(x_train, dtype=torch.float)
            label_train = torch.tensor(y_train, dtype=torch.float)

            # optimize
            for t in range(num_epochs):
                # Forward pass: Compute predicted y by passing x to the model
                y_pred = net(X_train)

                # Compute the loss
                loss = criterion(y_pred, label_train)

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Finally, the learned FMs
            FM_learned = (net.chi_nn_vars(net.vars).cpu()).detach().numpy()

            # print("FM_learned")
            # print(FM_learned)
            successes, total = self.test(net, x_test, y_test)

            elapsed = time.time() - start
            cum_time += elapsed
            cum_succ.append([successes, total])
            print("Elapsed time last loop:", format_timespan(elapsed),
                  "Total elapsed time:", format_timespan(cum_time))
            print(successes, "successes from ", len(x_test), " success rate =",
                  successes * 100 / len(x_test), "% Accumulated success rate:",
                  np.average([x / y for x, y in cum_succ]) * 100, "% in", len(cum_succ), "runs")

        print("Average success rate for", splits, "runs:",
              np.average([x / y for x, y in cum_succ]) * 100, "% - in", format_timespan(cum_time))


class Classificator():
    """
    A classificator to test Yi-Chung Hu's article
    It learns alone fuzzy measures and estimates lambda to perform 
    a Choquet Integral
    """

    def __init__(self, FI):
        self.ds = Datasets()

        # Algorithm parameters
        # 5.2. Pre-specified parameter specifications
        self.NPOP = 50  # Population size
        self.NCON = 500  # Total number of generations
        self.Ndel = 2  # for not generating much perturbation in the next generation,
        # a small number of the elite chromosomes is taken into account.
        self.IND_SIZE = 10  # for 3 decimal precision in binary string
        self.Prc = 0.95  # 0.5 deap example. works better
        self.Prm = 0.05  # 0.2 deap example. works better
        # Evaluation function parameters
        self.wca = 1.0
        self.we = 0.1
        # CFISLP is handled as an individual consisting of (n + 1) substrings
        # NATTR = len(x[0])+1 # +1 for the cut parameter

        self.FI = FI

    def test(self, individual, x_test, y_test):
        floats = utils.individualtofloat(individual)
        weights = floats[:-1]
        cutvalue = floats[-1]
        FI = FIntegrals()
        l = utils.bisectionLien(FI.FLambda, np.array(weights))
        successes = 0
        for x_testi, y_testi in zip(x_test, y_test):
            CFI = self.FI(x_testi, weights, l)  # With landa less expensive testing...
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
                  successes * 100 / len(x_test), "% Accumulated success rate:",
                  np.average([x / y for x, y in cum_succ]) * 100, "% in", len(cum_succ), "runs")

        print("Average success rate for", splits, "runs:",
              np.average([x / y for x, y in cum_succ]) * 100, "% - in", format_timespan(cum_time))

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
                  successes * 100 / len(x_test), "% Accumulated success rate:",
                  np.average([x / y for x, y in cum_succ]) * 100, "% in", len(cum_succ), "runs")

        print("Average success rate for", times, "runs:",
              np.average([x / y for x, y in cum_succ]) * 100, "% - in", format_timespan(cum_time))

    def NFolds(self, x, y, splits=10, repetitions=10):
        kf = KFold(n_splits=splits, shuffle=True)
        stats = utils.LogStats()
        for rep in range(repetitions):
            for train_index, test_index in kf.split(x):
                stats.startLogging()

                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                iFISLP = FISLP.FISLP(self.NPOP, self.NCON, self.Ndel, self.IND_SIZE,
                                     self.Prc, self.Prm, self.wca, self.we, x_train, y_train, self.FI)
                best, pop, log = iFISLP.GAFISLP()
                successes, total = self.test(best[0], x_test, y_test)

                stats.update(successes, total)
                stats.printStats()

            stats.printRepetitionStats()
            stats.newRepetition()
        stats.printFinalStats()

        return stats.average(), stats

    def NDivide(self, x, y, size=5, times=10):
        stats = utils.LogStats()
        for i in range(times):
            stats.startLogging()

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size)

            iFISLP = FISLP.FISLP(self.NPOP, self.NCON, self.Ndel, self.IND_SIZE,
                                 self.Prc, self.Prm, self.wca, self.we, x_train, y_train, self.FI)
            best, pop, log = iFISLP.GAFISLP()
            successes, total = self.test(best[0], x_test, y_test)

            stats.update(successes, total)
            stats.printStats()

        stats.printRepetitionStats()

        return stats.average(), stats


# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting

def run_all():
    FI = FIntegrals()
    ds = Datasets()
    fitness_functions = [["Choquet Lambda", FI.ChoquetLambda],
                        ["Sugeno Lambda", FI.SugenoLambda],
                        ["Min Choquet Lambda", FI.MinChoquetLambda]]
    appen = ["Appendicitis", ds.Appendicitis]
    thyroid = ["Thyroid", ds.Thyroid]
    datasets = [["GermanCredit", ds.GermanCredit],
                ["Breast Cancer Wisconsin", ds.BreastCancerWisconsin],
                ["Breast Cancer", ds.BreastCancer],
                ["Diabetes", ds.Diabetes],
                ["Cleveland HD (Kaggle)", ds.KaggleClevelandHD],
                ["Heart Disease 2", ds.StatLogHeart2]]

    for F in fitness_functions:
        print("------------------ {} ------------- ".format(F[0]))
        for data in appen+datasets:
            print("------------------ {} ------------- ".format(data[0]))
            classi = Classificator(F)
            x, y = data[1]()
            if data[0] == "Appendicitis":
                avg, stats = classi.NFolds(x, y, splits=len(y), repetitions=2)
            elif data[0] == "Breast Cancer":
                av, stats = classi.NDivide(x, y)
            elif data[0] == "Thyroid":
                pass
            else:
                avg, stats = classi.NFolds(x, y, repetitions=1)


from functools import partial


def run_bcw():
    ds = Datasets()
    FI = FIntegrals()
    fitness_functions = [["Choquet Lambda", FI.ChoquetLambda],
                         ["Sugeno Lambda", FI.SugenoLambda],
                         ["Generalized Sugeno Lambda Sum of products",
                          partial(FI.GeneralizedSugenoLambda, lambda x, y: x * y, sum)],
                        ["Min Choquet Lambda", FI.MinChoquetLambda]]

    datasets = [["Breast Cancer Wisconsin", ds.BreastCancerWisconsin]]

    for F in fitness_functions:
        print("------------------ {} ------------- ".format(F[0]))
        for data in datasets:
            print("------------------ {} ------------- ".format(data[0]))
            classi = Classificator(F[1])
            x, y = data[1]()
            avg, stats = classi.NFolds(x, y, repetitions=1)
            print(avg)
            print(stats)


def run_choquet_nn():
    print("------------------ Choquet NN ------------- ")
    print("----------------- Breast Cancer Wisconsin------------ ")
    cticlas = CITClassificator()
    x, y = cticlas.ds.BreastCancerWisconsin()
    cticlas.Folds(x, y)


if __name__ == "__main__":
    run_bcw()
    # run_all()
