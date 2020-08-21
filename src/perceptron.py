#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 22:50:34 2020

@author: asier
"""
import torch
import numpy as np
import torch.nn as nn

import time
from humanfriendly import format_timespan
from sklearn.model_selection import KFold
from concurrent.futures import  ProcessPoolExecutor


from Datasets import Datasets


class SLP(nn.Module):  

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        return x

    def predict(self, x):
        x = torch.sigmoid(self.linear(x))
        return 1 if x >= 0.5 else 0


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
        return 1 if x >= 0.5 else 0

def ptrain(x, y, train_index, test_index, M):
    start = time.time()
    successes = 0
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    xdata = torch.Tensor(x_train)
    ydata = torch.Tensor(y_train.reshape(len(xdata), 1))
    torch.manual_seed(2)
    model = M(len(xdata[0]), 1)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.9, momentum=0.5)
    epochs = 10000
    losses = []
    for i in range(epochs):
        ypred = model.forward(xdata)
        loss = criterion(ypred, ydata)
        #print("epoch:", i, "loss:", loss.item())
        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    successes, total = test(x_test, y_test, model)

    elapsed = time.time() - start
    print("Elapsed time last loop:", format_timespan(elapsed))
    print(successes, "successes from ", len(x_test), " success rate =",
          successes*100/len(x_test), "%")
    return successes, total

def pFolds(x, y, M, splits=10):
    kf = KFold(n_splits=splits)
    cum_succ = []
    ind = list(kf.split(x))
    train = ind[0]
    test = ind[1]
    with ProcessPoolExecutor(max_workers=6) as executor:
        executor.map(ptrain, x, y, train, test, M)


def Folds(x, y, M, splits=10):
    kf = KFold(n_splits=splits)
    cum_time = 0
    cum_succ = []
    for train_index, test_index in kf.split(x):
        start = time.time()
        successes = 0
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        xdata = torch.Tensor(x_train)
        ydata = torch.Tensor(y_train.reshape(len(xdata), 1))
        torch.manual_seed(2)
        model = M(len(xdata[0]), 1)
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.9, momentum=0.5)
        epochs = 10000
        losses = []
        for i in range(epochs):
            ypred = model.forward(xdata)
            loss = criterion(ypred, ydata)
            #print("epoch:", i, "loss:", loss.item())
            losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        successes, total = test(x_test, y_test, model)

        elapsed = time.time() - start
        cum_time += elapsed
        cum_succ.append([successes, total])
        #print("Elapsed time last loop:",format_timespan(elapsed),"Total elapsed time:",format_timespan(cum_time))
        #print(successes,"successes from ",len(x_test)," success rate =",successes*100/len(x_test),"% Cum:",np.average([x/y for x,y in cum_succ])*100,"% ")

    print("Average success rate for", splits, "runs:",
          np.average([x/y for x, y in cum_succ])*100, "% - in", format_timespan(cum_time))



def test(x_test, y_test, model, epsilon=0.001):
    xdata = torch.Tensor(x_test)
    ydata = torch.Tensor(y_test.reshape(len(xdata), 1))
    succ = 0
    for x, y in zip(xdata, ydata):
        if abs(model.predict(x) - y) < epsilon:
            succ += 1
    return succ, len(x_test)



ds = Datasets()
print(" ----- SLP ---- ")
print("-- Appendicitis -- ")
x, y = ds.Appendicitis()
Folds(x, y, SLP, len(y))
print("-- Breast Cancer -- ")
x, y = ds.BreastCancer()
Folds(x, y, SLP)
print("-- Breast Cancer Wisconsin -- ")
x, y = ds.BreastCancerWisconsin()
Folds(x, y, SLP)
print("-- Diabetes -- ")
x, y = ds.Diabetes()
Folds(x, y, SLP)
print("-- Cleveland HD -- ")
x, y = ds.ProcessedClevelandHD()
Folds(x, y, SLP)
print("-- Heart Disease -- ")
x, y = ds.StatlogHeart()
Folds(x, y, SLP)
print("-- GermanCredit -- ")
x, y = ds.GermanCredit()
Folds(x, y, SLP)
print("-- Thyroid -- ")
x, y = ds.Thyroid()
Folds(x, y, SLP)

print(" ----- MLP ---- ")
print("-- Appendicitis -- ")
x, y = ds.Appendicitis()
Folds(x, y, MLP, len(y))
print("-- Breast Cancer -- ")
x, y = ds.BreastCancer()
Folds(x, y, MLP)
print("-- Breast Cancer Wisconsin -- ")
x, y = ds.BreastCancerWisconsin()
Folds(x, y, MLP)
print("-- Diabetes -- ")
x, y = ds.Diabetes()
Folds(x, y, MLP)
print("-- Cleveland HD -- ")
x, y = ds.ProcessedClevelandHD()
Folds(x, y, MLP)
print("-- Heart Disease -- ")
x, y = ds.StatlogHeart()
Folds(x, y, MLP)
print("-- GermanCredit -- ")
x, y = ds.GermanCredit()
Folds(x, y, MLP)
print("-- Thyroid -- ")
x, y = ds.Thyroid()
Folds(x, y, MLP)
