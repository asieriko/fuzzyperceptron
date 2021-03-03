#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 12:02:33 2020

@author: asier_urio
"""

import csv
import os
from functools import partial

import numpy as np
# Perceptron learning algorithm using PyTorch
import torch
import torch.nn as nn

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

from Datasets import Datasets, Folds
from utils import LogStats

"""
Function definitions for the generalized adaline operator
the following functions need to be adjusted to work along the axis
max = torch.max
min = torch.min
avg = torch.mean
sum = torch.sum
Move this functions to other file?
"""


def absdot(x, o):
    return torch.abs(o) * x


def sqrdot(x, o):
    return (o ** 2) * x


def dot(y, w):
    return y * w
    # return [y*wi for wi in w] #For more than one output


def cavg(y, w):
    return torch.true_divide(y + w, 2)


def sgnavg(y, w):
    return torch.sign(y * w) * torch.true_divide(torch.abs(y) + torch.abs(w), 2)


def sgnmax(y, w):
    return torch.sign(y * w) * torch.max(torch.abs(y), torch.abs(w))


def sgnmin(y, w):
    return torch.sign(y * w) * torch.min(torch.abs(y), torch.abs(w))


"""
In the next functions there is a call to view to avoid
the following warning:
    loss.py:516: UserWarning: Using a target size (torch.Size([105, 1])) 
    that is different to the input size (torch.Size([105])) is deprecated.
"""


def mmin(x):
    return torch.min(x, dim=1, keepdim=True)  #
    # return torch.min(x,axis=1).values.view(len(x),1)


def mmax(x):
    return torch.max(x, dim=1, keepdim=True)


def msum(x):
    y = torch.sum(x, dim=1, keepdim=True)
    return y


def mmean(x):
    return torch.mean(x, dim=1, keepdim=True)


def hamacher(x, y, epsilon=1e-10):
    print("hx", x)
    print("hy", y)
    return x * y / (epsilon + x + y - x * y)  # epsilon to avoid 0/0 nan
    # For multiple layers


# adaptadas a multilayer

# F's
def mmean2(x):
    return torch.mean(x, dim=2)


def mmax2(x):
    return torch.max(x, dim=2)


def mmin2(x):
    return torch.min(x, dim=2)


def msum2(x):
    return torch.sum(x, dim=2)


# B's
def madd(x, y):
    return torch.add(x.unsqueeze(1), y)


# g's

def mdot(x, y):
    xu = x.unsqueeze(1)
    xy = torch.mul(xu, y)
    return xy


def mhamacher(x, y, epsilon=1e-10):
    xu = x.unsqueeze(1)
    xy = torch.mul(xu, y)
    return xy / (epsilon + xu + y - xy)  # espsilon to avoid 0/0 nan
    # torch.mul(x.unsqueeze(1),y)/(x.unsqueeze(1) + y - torch.mul(x.unsqueeze(1),y)


def m2min(x, y):
    return torch.min(x.unsqueeze(1), y)


def m2max(x, y):
    return torch.max(x.unsqueeze(1), y)


def mavg(x, y):
    xu = x.unsqueeze(1)
    return torch.true_divide(xu + y, 2)


def t_luckas(x, y):
    m = torch.zeros_like(y)
    xu = x.unsqueeze(1)
    return torch.max(m, xu + y - 1)


def mgeom(x, y):
    xu = x.unsqueeze(1)
    xy = torch.mul(xu, y)
    return torch.sqrt(xy)


def probsum(x, y):
    # return x+y-x*y
    xu = x.unsqueeze(1)
    xy = torch.mul(xu, y)
    return xu + y - xy


def c_luckas(x, y):
    xu = x.unsqueeze(1)
    m = torch.ones_like(y)
    return torch.min(m, xu + y)


def oxy(x, y):
    xu = x.unsqueeze(1)
    return torch.min(torch.sqrt(xu) * y, xu * torch.sqrt(y))


# Prueba inicial
# F = [mmin,msum,mmean,mmax]
# B = [torch.add,cavg,torch.max,torch.min,absdot,sqrdot]
# g = [dot,sgnavg,sgnmax,sgnmin]

# Prueba 2. Fijar F y B como + y variar g
F = [msum2]
B = [torch.add]
g = [mdot, m2min, mavg, t_luckas, mgeom, oxy, m2max, probsum, c_luckas]  # mhamacher


def fGADALINE(x, w, t, g, B, F):
    return F(B(g(x, w), t))


class NewGADALINE(nn.Module):
    """
    Generalized Adaline Operator
    :math:`P_{\\vec{\\omega}, \\vec{\\theta}}^{g,B,F} (x_1,...,x_n) = F(B_1(g_1(\\omega_1,x_1),\\theta_1),...,B_n(g_n(\\omega_n,x_n),\\theta_n))`.
    """

    def __init__(self, g, B, F, input_size, output_size):
        super().__init__()
        self.num_features = input_size
        self.weight = nn.Parameter(torch.zeros(input_size, output_size,
                                               dtype=torch.float), requires_grad=True)
        # Addedd output size in bias for hidden layers
        # self.bias = nn.Parameter(torch.zeros(input_size,output_size, dtype=torch.float),requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(input_size, output_size, dtype=torch.float), requires_grad=True)
        self.g = g
        self.B = B
        self.F = F

    def forward(self, x):
        output = fGADALINE(x, self.weight, self.bias, self.g, self.B, self.F)
        output = torch.sigmoid(output)
        return output


class GADALINE(nn.Module):
    """
    Generalized Adaline Operator
    :math:`P_{\\vec{\\omega}, \\vec{\\theta}}^{g,B,F} (x_1,...,x_n) = F(B_1(g_1(\\omega_1,x_1),\\theta_1),...,B_n(g_n(\\omega_n,x_n),\\theta_n))`.
    """

    def __init__(self, g, B, F, input_size, output_size):
        super().__init__()
        self.num_features = input_size
        self.weight = nn.Parameter(torch.zeros(input_size, output_size,
                                               dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(input_size, output_size, dtype=torch.float), requires_grad=True)
        self.g = g
        self.B = B
        self.F = F

    def forward(self, x):
        output = self.g(x, self.weight.t())
        output = self.B(output, self.bias.t())
        output = self.F(output)
        output = torch.sigmoid(output)
        return output

    def _sign(self, x):
        return 0 if x >= 0.5 else 1.
        return 1. if x >= 0 else -1.

    def predict(self, x):
        x = self.forward(x).detach()
        return x


def mse(pred, target):
    return (pred - target) ** 2


class SLP(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        # Parameters initialization, same as gadaline
        # self.linear.weight.data.fill_(0)
        # self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        # m = nn.ReLU()
        # x = m(x)
        return x

    def predict(self, x):
        x = torch.sigmoid(self.linear(x))
        return x


class MLP(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.hidden = nn.Linear(input_size, 3)
        self.output = nn.Linear(3, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

    def predict(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x


class MLGADALINE(nn.Module):

    def __init__(self, g, B, F, input_size, output_size):
        super().__init__()
        self.hidden = GADALINE(g, B, F, input_size, 3)
        self.output = GADALINE(g, B, F, 3, output_size)

    def forward(self, x):
        # print("X",x)
        # x = torch.sigmoid(self.hidden(x)) ADALINES does sigmoid
        x = self.hidden(x)
        # print("hidden",x)
        # x = torch.sigmoid(self.output(x))  ADALINES does sigmoid
        x = self.output(x)
        # print("out",x)
        return x

    def predict(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x
        # return 1 if x >= 0.5 else 0


def NNtrain(xdata, ydata, model):
    """
    This funcion trains and test provided data under the given model M

    Parameters
    ----------
    xdata : torch.tensor
        dataset's values
    ydata : torch.tensor
        dataset's expected results
    model : torch.nn model
        The model to perform the training with

    Returns
    -------
    model : PyTorch model
        Trained model.
    """
    criterion = nn.MSELoss()
    # Original 0.9 -> optimizer = torch.optim.SGD(model.parameters(), lr=0.9, momentum=0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.9, momentum=0.5)
    epochs = 10000
    losses = []
    for i in range(epochs):
        # for xt,yt in zip(xdata,ydata):
        xt, yt = xdata, ydata
        ypred = model(xt)
        loss = criterion(ypred, yt)
        # loss.requres_grad = True #FIXME: It worked befor without this¿?
        losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def bFolds(x, y, train_index, test_index, M):
    """
    This funcion trains and test provided data under the given model M

    Parameters
    ----------
    x : torch.tensor
        dataset's values
    y : torch.tensor
        dataset's expected results
    train_index : list
        input index to use as train data
    test_index : torch.tensor
        input index to use as test data
    M : torch.nn model
        The model to perform the training with

    Returns
    -------
    model : PyTorch model
        Trained model.
    succeses : integer
        amount of training instances correctly clasified
    total : integer
        total training instances
    stats : statslog object with a summary of accuracies
        and related data
    """
    model = M(len(x[0]), 1)

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    xdata = torch.Tensor(x_train)
    ydata = torch.Tensor(y_train.reshape(len(xdata), 1))

    model = NNtrain(xdata, ydata, model)

    successes, total = test(x_test, y_test, model)

    return model, successes, total


def trainTest(x_train, y_train, x_test, y_test, M):
    """
    This funcion trains and test provided data under the given model M
    Parameters
    ----------
    x_train : torch.tensor
        train dataset's values
    y_train : torch.tensor
        train dataset's expected results
    x_test : torch.tensor
        test dataset's values
    y_test : torch.tensor
        test dataset's expected results
    M : torch.nn model
        The model to perform the training with

    Returns
    -------
    model : PyTorch model
        Trained model.
    stats : statslog object with a summary of accuracies
        and related data
    """
    model = M(len(x_train[0]), 1)
    xdata = torch.Tensor(x_train)
    ydata = torch.Tensor(y_train.reshape(len(xdata), 1))

    stats = LogStats()

    model = NNtrain(xdata, ydata, model)
    successes, total = test(x_test, y_test, model)

    stats.update(successes, total)
    # stats.printStats()
    #
    # stats.printRepetitionStats()
    # stats.printFinalStats()
    return model, stats


def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(.5).view(-1)
    expected = torch.Tensor(y_true).eq(1).view(-1)
    return (expected == predicted).sum().float(), len(y_true)


def test(x_test, y_test, model):
    """
    This funcion test provided data against the given model M

    Parameters
    ----------
    x_test : torch.tensor
        dataset's values
    y_test : torch.tensor
        dataset's expected results
    model : torch.nn model
        The model to perform the test with

    Returns
    -------
    succ : integer
        amount of succesfully classified instances
    total : integer
        total instances
    """
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
    # https://curiousily.com/posts/build-your-first-neural-network-with-pytorch/
    y_test_pred = model(torch.Tensor(x_test))
    y_test_pred = torch.squeeze(y_test_pred)
    succ, total = calculate_accuracy(y_test, y_test_pred)
    return succ, total


ds = Datasets()

run = "-folds-25-02-2021"

datasets = [["GermanCredit", ds.GermanCredit, 75.4, 73.1],
            ["Breast Cancer W", ds.BreastCancerWisconsin, 96.1, 96.5],
            ["Breast Cancer", ds.BreastCancer, 71.3, 75.3],
            ["Statlog HD", ds.StatLogHeart2, 84.5, 82.9],
            ["Appendicitis", ds.Appendicitis, 85.8, 85.8]]

datasetsnoonline = [["Diabetes", ds.Diabetes, 73.6, 76.8],
                    ["Cleveland HD ", ds.KaggleClevelandHD, 83.5, 82.1]]
# dsthyroid = ["Thyroid",ds.Thyroid,97.4,96.2]
datasets += datasetsnoonline


def train_dataset(x, y, model, k=10, r=10):
    avg = []
    for i in range(r):
        avgfold = []
        kf = KFold(n_splits=k, shuffle=True)
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            xdata = torch.Tensor(x_train)
            ydata = torch.Tensor(y_train.reshape(len(xdata), 1))

            model = NNtrain(x_train, y_train, model)
            suc, total = test(x_test, y_test, model)

            avgfold.append(suc / total)
        avg.append(avgfold)

    return np.average(avg)


def testGADALINE():
    for model in ["SLP", "MLP"]:
        ptm = SLP if model == "SLP" else MLP
        for t in datasets:
            xd = []
            yd = []
            print("-- " + t[0] + " -- ")
            print("------ {}: Base -----".format(model))
            plotname = t[0] + " - " + model
            x, y = t[1]()
            avg = []
            for i in range(10):
                avgfold = []
                for train_index, test_index in Folds(t[0],i):
                    m, suc, total = bFolds(x, y, train_index, test_index, ptm)
                    avgfold.append((suc / total).tolist())
                avg.append(avgfold*100)
                print(t[0],i,avgfold)
            # print(avg)
            print("{}: {} repetitions: {}% accuracy".format(t[0],i+1,np.average(avg)))
            with open(os.path.join("..", "Results", t[0] + run + ".csv"), "a") as f:
                fwriter = csv.writer(f, delimiter=",")
                fwriter.writerow([t[0], model, "", "", "", np.average(avg)])
            plt.show()
            xd.append(model)
            yd.append(np.average(avg))

    print("------ MGADALINE -----")

    # print("-- Comprobar: g = *, B = +, F = + -- ")
    # gadaline = partial(GADALINE,dot,torch.add,msum)
    # m, stats = Folds(x, y, gadaline)#, len(y))
    # with open(t[0]+run+".csv", "a") as f:
    #     fwriter = csv.writer(f,delimiter=",")
    #     fwriter.writerow([t[0],"GADALINE","dot","add","msum",stats.average(),stats.elapsed()])
    for t in datasets:
        x, y = t[1]()
        print("-- Functions -- ")
        for gi in g:
            for Bi in B:
                for Fi in F:
                    print("gi = {}\tBi = {}\tFi = {}".format(gi.__name__, Bi.__name__, Fi.__name__))
                    plotname = t[0] + " - gi = {}\nBi = {}\nFi = {}\n".format(gi.__name__, Bi.__name__, Fi.__name__)
                    gadaline = partial(MLGADALINE, gi, Bi, Fi)
                    avg = []
                    for i in range(10):
                        avgfold = []
                        for train_index, test_index in Folds(t[0],i):
                            x_train, x_test = x[train_index], x[test_index]
                            y_train, y_test = y[train_index], y[test_index]
                            # _, sucpt, totpt = bFolds(x_train, y_train, x_test, y_test, m)
                            m, suc, total = bFolds(x, y, train_index, test_index, gadaline)
                            avgfold.append((suc*100 / total).tolist())
                        avg.append(avgfold)
                        print(t[0], i, avgfold)
                        with open(os.path.join("..", "Results", t[0] + run + ".csv"), "a") as f:
                            fwriter = csv.writer(f, delimiter=",")
                            fwriter.writerow(
                                [t[0], "M3GADALINE", gi.__name__, Bi.__name__, Fi.__name__, np.average(avg)])
                    print("{}: {} repetitions: {}% accuracy".format(t[0], i + 1, np.average(avg)))


def thyroid():
    print(" ----- SLP: Base ---- ")
    print("-- Thyroid -- ")
    x_train, y_train, x_test, y_test = ds.Thyroid()
    m, stats = trainTest(x_train, y_train, x_test, y_test, SLP)
    with open("resultsThyroidN.csv", "a") as f:
        fwriter = csv.writer(f, delimiter=",")
        fwriter.writerow(["Thyroid", "SLP", ",""", "", stats.average(), stats.elapsed()])

    for gi in g:
        for Bi in B:
            for Fi in F:
                print("\n gi = {}\nBi = {}\nFi = {}".format(gi.__name__, Bi.__name__, Fi.__name__))
                gadaline = partial(GADALINE, gi, Bi, Fi)
                m, stats = trainTest(x_train, y_train, x_test, y_test, gadaline)
                with open(os.path.join("..", "Results", "resultsThyroidN" + run + ".csv"), "a") as f:
                    fwriter = csv.writer(f, delimiter=",")
                    fwriter.writerow(
                        ["Thyroid", "GADALINE", gi.__name__, Bi.__name__, Fi.__name__, stats.average(),
                         stats.elapsed()])


if __name__=="__main__":
    testGADALINE()