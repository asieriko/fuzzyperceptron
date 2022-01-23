#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 09:40:42 2021

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

from sklearn.model_selection import KFold

from Datasets import Datasets
import matplotlib.pyplot as plt


def my_plot(epochs, loss):
    plt.plot(epochs, loss)
    
class SLP(nn.Module):  

    def __init__(self, input_size, output_size,f=nn.ReLU()):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        #Parameters initialization, same as gadaline
        self.linear.weight.data.fill_(0)
        self.linear.bias.data.fill_(0)
        self.f = f

    def forward(self, x):
        x = self.linear(x)
        x = self.f(x)
        return x

    def predict(self, x):
        x = self.f(self.linear(x))
        return x

def ptrain(xdata, ydata, model):
    criterion = nn.MSELoss()
    #Original 0.9 -> optimizer = torch.optim.SGD(model.parameters(), lr=0.9, momentum=0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.09, momentum=0.5)
    epochs = 10000
    losses = []
    xplot = []
    yplot = []
    for i in range(epochs):
        #for xt,yt in zip(xdata,ydata):
         xt,yt = xdata,ydata
         ypred = model(xt)
         loss = criterion(ypred, yt)
         #loss.requres_grad = True #FIXME: It worked befor without this¿?
         xplot.append(i)
         yplot.append(loss)
         losses.append(loss)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         my_plot(xplot,yplot)

    return model


def bFolds(x, y, M, splits=10):
    kf = KFold(n_splits=splits)
    cum_time = 0
    cum_succ = []
    model = M(len(x[0]), 1)
    for train_index, test_index in kf.split(x):
        start = time.time()
        successes = 0
        
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        xdata = torch.Tensor(x_train)
        ydata = torch.Tensor(y_train.reshape(len(xdata), 1))
        
        model = ptrain(xdata, ydata, model)
        
        successes, total = test(x_test,y_test,model)
        
        cum_time +=  time.time() - start
        cum_succ.append([successes, total])
        #print("Elapsed time last loop:",format_timespan(elapsed),"Total elapsed time:",format_timespan(cum_time))
        #print(successes,"successes from ",len(x_test)," success rate =",successes*100/len(x_test),"% Cum:",np.average([x/y for x,y in cum_succ])*100,"% ")
    av = np.average([x/y for x, y in cum_succ])*100
    ds = np.std([x/y for x, y in cum_succ])
    ti = cum_time
    print("Average success rate for", splits, "runs:",
          np.average([x/y for x, y in cum_succ])*100, "% +/-" + str(np.std([x/y for x, y in cum_succ])) + " - in", format_timespan(cum_time))
    return model, av, ds, ti


def trainTest(x_train,y_train,x_test,y_test,M):
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
    
    successes = 0
    start = time.time()
    
    model = ptrain(xdata,ydata,model)
    successes, total = test(x_test,y_test,model)
    
    av = successes*100/total
    ds = 0
    ti = time.time() - start
    print("Model Accuracy:", av, " - in", format_timespan(ti))
    return model, av, ds, ti

def calculate_accuracy(y_true, y_pred):
  predicted = y_pred.ge(.5).view(-1)
  expected = torch.Tensor(y_true).eq(1).view(-1)
  return (expected == predicted).sum().float(), len(y_true)

def test(x_test, y_test, model):
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
    #https://curiousily.com/posts/build-your-first-neural-network-with-pytorch/
    y_test_pred = model(torch.Tensor(x_test))                    
    y_test_pred = torch.squeeze(y_test_pred)
    succ, total = calculate_accuracy(y_test, y_test_pred)
    return succ, total
    
ds = Datasets()

# afunctions = [nn.ELU,nn.Hardshrink,nn.Hardsigmoid,nn.Hardtanh,nn.Hardswish,
#              nn.LeakyReLU,nn.LogSigmoid,nn.MultiheadAttention,nn.PReLU,
#              nn.ReLU,nn.ReLU6,nn.RReLU,nn.SELU,nn.CELU,nn.GELU,nn.sigmoid,
#              nn.SiLU,nn.Softplus,nn.Softshrink,nn.Softsign,nn.Tanh,
#              nn.Tanhshrink,nn.Thresold,nn.Softmin,nn.Softmax,nn.Softmax2d,
#              nn.LogSoftmax,nn.AdaptiveLogSoftmaxWithLoss]

# lfunctions = [nn.L1Loss,nn.MSELoss,nn.CrossEntropyLoss,nn.CTCLoss,nn.NLLLoss,
#               nn.PoissonNLLLoss,nn.KLDivLoss,nn.BCELoss,nn.BCEWithLogitsLoss,
#               nn.MarginRankingLoss,nn.HingeEmbedingLoss,nn.MultiLabelMarginLoss,
#               nn.SmoothL1Loss,nn.SoftMarginLoss,nn.MultiLabelSoftMarginLoss,
#               nn.CosineEmbeddingLoss,nn.MultiMarginLoss,nn.TripletMarginLoss,
#               nn.TripletMarginWithDistanceLoss]

tests = [["Appendicitis",ds.Appendicitis],
         ["GermanCredit",ds.GermanCredit],
         ["Breast Cancer Wisconsin",ds.BreastCancerWisconsin],
         ["Breast Cancer", ds.BreastCancer],
         ["Diabetes",ds.Diabetes],
         ["Cleveland HD",ds.ProcessedClevelandHD],
         ["Heart Disease",ds.StatlogHeart],
         ["Thyroid",ds.Thyroid]]


#SLPx = partial(SLP,nn.ReLU)
x, y = ds.GermanCredit()
bFolds(x,y,SLP)

asdf
# Global iteration

for t in tests:
    print("-- " + t[0] + " -- ")
    print(" ----- SLP: Base ---- ")
    
    x, y = t[1]()
    m, av, sd, ti = bFolds(x, y, SLP)#, len(y))
    with open(os.path.join("..","Results", t[0]+"-test.csv"), "a") as f:
        fwriter = csv.writer(f,delimiter=",")
        fwriter.writerow([t[0],"SLP","","","",av,sd,ti])