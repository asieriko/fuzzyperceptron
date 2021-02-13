#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 12:02:33 2020

@author: asier_urio
"""

import time
import csv
import os
from humanfriendly import format_timespan
from functools import partial

import numpy as np
# Perceptron learning algorithm using PyTorch
import torch
import torch.nn as nn

# from humanfriendly import format_timespan
from sklearn.model_selection import KFold
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt

from Datasets import Datasets
from utils import LogStats

plotname = ""


def my_plot(epochs, loss):
    # print("Ploting")
    plt.plot(epochs, loss)
    plt.title(plotname)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")


def my_plot2(x, y, name):
    # print("Ploting")
    plt.bar(x, y)
    tics = [i for i in range(len(x))]
    plt.title(name)
    plt.xticks(tics, x, rotation='vertical')
    plt.xlabel("functions")
    plt.ylabel("accuracy")
    plt.savefig("/home/asier/Hezkuntza/Ikerketa/Thesis/Results/" + name + run + ".png")
    plt.show()


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


def hamacher(x, y):
    print("hx", x)
    print("hy", y)
    return x * y / (x + y - x * y)  # FIXME: xxx
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


def mhamacher(x, y):
    xu = x.unsqueeze(1)
    xy = torch.mul(xu, y)
    return xy / (xu + y - xy)
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


# mhamacher genera 0/0 -> nan cuando un valor del input es 0 y w tb
# Chapuceramente se puede hacer
# y=0.000001
# x+y-x*y
# en los datos de entrada, así no hay 0s


def fGADALINE(x, w, t, g, B, F):
    return F(B(g(x, w), t))


class NewGADALINE(nn.Module):
    '''
    Generalized Adaline Operator
    :math:`P_{\\vec{\\omega}, \\vec{\\theta}}^{g,B,F} (x_1,...,x_n) = F(B_1(g_1(\\omega_1,x_1),\\theta_1),...,B_n(g_n(\\omega_n,x_n),\\theta_n))`.
    '''

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
    '''
    Generalized Adaline Operator
    :math:`P_{\\vec{\\omega}, \\vec{\\theta}}^{g,B,F} (x_1,...,x_n) = F(B_1(g_1(\\omega_1,x_1),\\theta_1),...,B_n(g_n(\\omega_n,x_n),\\theta_n))`.
    '''

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
        self.linear.weight.data.fill_(0)
        self.linear.bias.data.fill_(0)

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
    criterion = nn.MSELoss()
    # Original 0.9 -> optimizer = torch.optim.SGD(model.parameters(), lr=0.9, momentum=0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.9, momentum=0.5)
    epochs = 10000
    losses = []
    yplot = []
    xplot = []
    for i in range(epochs):
        # for xt,yt in zip(xdata,ydata):
        xt, yt = xdata, ydata
        ypred = model(xt)
        loss = criterion(ypred, yt)
        # loss.requres_grad = True #FIXME: It worked befor without this¿?
        losses.append(loss)

        if i % 100 == 0:
            xplot.append(i)
            yplot.append(loss)
            my_plot(xplot, yplot)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def bFolds(x, y, M, splits=10):
    kf = KFold(n_splits=splits, shuffle=True)
    # kf = RepeatedKFold(n_splits=2, n_repeats=2)
    stats = LogStats()
    model = M(len(x[0]), 1)
    for train_index, test_index in kf.split(x):
        stats.startLogging()

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        xdata = torch.Tensor(x_train)
        # #FIXME avoid 0/0 in hamacher
        # eps=0.000001
        # xdata = xdata+eps-xdata*eps
        ydata = torch.Tensor(y_train.reshape(len(xdata), 1))

        model = NNtrain(xdata, ydata, model)

        successes, total = test(x_test, y_test, model)

        stats.update(successes, total)
        # stats.printStats()

    # stats.printRepetitionStats()
    # stats.printFinalStats()
    return model, stats


def trainTest(x_train, y_train, x_test, y_test, M):
    """
    This funcion trains and test provided data under the given model M
    Parameters
    ----------
    x_train : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    x_test : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    M : TYPE
        DESCRIPTION.

    Returns
    -------
    model : PyTorch model
        Trained model.
    av : float (0-100)
        Accuracy: Average successes in the train data.
    ds : TYPE
        Standar deviation (not aplicable in this case because only one 
                           train/test cicle is done""
    ti : float
        Elapsed time (train+test).

    """
    model = M(len(x_train[0]), 1)
    xdata = torch.Tensor(x_train)
    ydata = torch.Tensor(y_train.reshape(len(xdata), 1))

    # #FIXME avoid 0/0 in hamacher
    # eps=0.000001
    # xdata = xdata+eps-xdata*eps

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
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
    # https://curiousily.com/posts/build-your-first-neural-network-with-pytorch/
    y_test_pred = model(torch.Tensor(x_test))
    y_test_pred = torch.squeeze(y_test_pred)
    succ, total = calculate_accuracy(y_test, y_test_pred)
    return succ, total


ds = Datasets()

run = "-MGAD-12-02-2021"

appen = ["Appendicitis", ds.Appendicitis]
thyroid = ["Thyroid", ds.Thyroid]
datasets = [["GermanCredit", ds.GermanCredit],
            ["Breast Cancer Wisconsin", ds.BreastCancerWisconsin],
            ["Breast Cancer", ds.BreastCancer],
            ["Diabetes", ds.Diabetes],
            ["Cleveland HD (Kaggle)", ds.KaggleClevelandHD],
            ["Heart Disease 2", ds.StatLogHeart2]]

bad = [["Cleveland HD", ds.ProcessedClevelandHD],
       ["Heart Disease", ds.StatlogHeart]]


# Global iteration

def testDatasetsSLP():
    for t in datasets:
        print("SLP Results for {} dataset:".format(t[0]), end=" ")
        x, y = t[1]()
        avg = []
        for i in range(10):
            m, stats = bFolds(x, y, SLP)
            avg.append(stats.average(0))
        print("Accuracy: {:.2f}".format(np.average(avg)))

    for t in datasets:
        print("MLP Results for {} dataset:".format(t[0]), end=" ")
        x, y = t[1]()
        avg = []
        for i in range(10):
            m, stats = bFolds(x, y, MLP)
            avg.append(stats.average(0))
        print("Accuracy: {:.2f}".format(np.average(avg)))

    # Appendicitis leave-one out
    print("SLP Results for Appendicitis dataset:", end=" ")
    x, y = ds.Appendicitis()
    m, stats = bFolds(x, y, SLP, len(y))
    print("Accuracy: {:.2f}".format(stats.average(0)))

    print("MLP Results for Appendicitis dataset:", end=" ")
    x, y = ds.Appendicitis()
    m, stats = bFolds(x, y, MLP, len(y))
    print("Accuracy: {:.2f}".format(stats.average(0)))

    # Thyroid train vs test data
    print("SLP Results for Thyroid dataset:", end=" ")
    x_train, y_train, x_test, y_test = ds.Thyroid()
    m, stats = trainTest(x_train, y_train, x_test, y_test, SLP)
    print("Accuracy: {:.2f}".format(stats.average(0)))

    print("MLP Results for Thyroid dataset:", end=" ")
    x_train, y_train, x_test, y_test = ds.Thyroid()
    m, stats = trainTest(x_train, y_train, x_test, y_test, MLP)
    print("Accuracy: {:.2f}".format(stats.average(0)))

    # Thyroid mix all data an 10-folds
    # print("SLP Results for Thyroid dataset:", end=" ")
    # x_train, y_train, x_test, y_test = ds.Thyroid()
    # x = np.concatenate((x_train, x_test))
    # y = np.concatenate((y_train, y_test))
    # avg = []
    # for i in range(10):
    #      m, stats = bFolds(x, y, SLP)
    #      avg.append(stats.average(0))
    # print("Accuracy: {:.2f}".format(np.average(avg)))
    #
    # print("MLP Results for Thyroid dataset:", end=" ")
    # avg = []
    # for i in range(10):
    #      m, stats = bFolds(x, y, SLP)
    #      avg.append(stats.average(0))
    # print("Accuracy: {:.2f}".format(np.average(avg)))


testDatasetsSLP()

end

for t in tests2:
    xd = []
    yd = []
    print("-- " + t[0] + " -- ")
    print(" ----- SLP: Base ---- ")
    plotname = t[0] + " - MLP"
    x, y = t[1]()
    avg = []
    for i in range(10):
        m, stats = bFolds(x, y, SLP)  # , len(y))
        avg.append(stats.average(0))
    print(avg)
    print(np.average(avg))
    with open(os.path.join("..", "Results", t[0] + run + ".csv"), "a") as f:
        fwriter = csv.writer(f, delimiter=",")
        fwriter.writerow([t[0], "SLP", "", "", "", stats.average(), stats.elapsed()])
    plt.show()
    xd.append("SLP")
    yd.append(stats.average())

    print(" ----- MLP 3: Base ---- ")
    # plotname = t[0] + " - MLP"
    x, y = t[1]()
    m, stats = bFolds(x, y, MLP)  # , len(y))
    with open(os.path.join("..", "Results", t[0] + run + ".csv"), "a") as f:
        fwriter = csv.writer(f, delimiter=",")
        fwriter.writerow([t[0], "MLP", "", "", "", stats.average(), stats.elapsed()])
    plt.show()
    xd.append("MLP")
    yd.append(stats.average())
    asfd
    print(" ----- MGADALINE ---- ")

    # print("-- Comprobar: g = *, B = +, F = + -- ")
    # gadaline = partial(GADALINE,dot,torch.add,msum)
    # m, stats = Folds(x, y, gadaline)#, len(y))
    # with open(t[0]+run+".csv", "a") as f:
    #     fwriter = csv.writer(f,delimiter=",")
    #     fwriter.writerow([t[0],"GADALINE","dot","add","msum",stats.average(),stats.elapsed()])

    print("-- Functions -- ")
    for gi in g:
        for Bi in B:
            for Fi in F:
                print("gi = {}\nBi = {}\nFi = {}\n".format(gi.__name__, Bi.__name__, Fi.__name__))
                plotname = t[0] + " - gi = {}\nBi = {}\nFi = {}\n".format(gi.__name__, Bi.__name__, Fi.__name__)
                gadaline = partial(MLGADALINE, gi, Bi, Fi)
                m, stats = bFolds(x, y, gadaline)  # , len(y))
                with open(os.path.join("..", "Results", t[0] + run + ".csv"), "a") as f:
                    fwriter = csv.writer(f, delimiter=",")
                    fwriter.writerow(
                        [t[0], "M3GADALINE", gi.__name__, Bi.__name__, Fi.__name__, stats.average(), stats.elapsed()])
                plt.show()
                xd.append("{}-{}-{}".format(gi.__name__, Bi.__name__, Fi.__name__))
                yd.append(stats.average())
    my_plot2(xd, yd, t[0])

s = input("Seguir? (S/N)")
if s != "S":
    quit()
### Global iteration ends

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
                    ["Thyroid", "GADALINE", gi.__name__, Bi.__name__, Fi.__name__, stats.average(), stats.elapsed()])
