#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 13:02:33 2021

@author: asier_urio
"""

import csv
import os
import math
from functools import partial

import numpy as np
# Perceptron learning algorithm using PyTorch
import torch
import torch.nn as nn

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

from Datasets import Datasets, Folds
import src.FIntegrals as FI
from src.utils import individualtofloat, bisectionLien
from src.GADALINE import bFolds
from utils import LogStats


class Choquet(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    TODO: Weights..
    """

    @staticmethod
    def forward(ctx, input, weight):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input,weight)
        landa = bisectionLien(FI.FLambda, weight.t().detach().numpy()[0])
        CFI = FI.ChoquetLambdaT(input, weight, landa)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input,weight, = ctx.saved_tensors
        print(grad_output)
        print(input)
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[input > 1] = 1
        # grad_input = grad_input.clamp_(min=0, max=1)
        return grad_input, weight



class CADALINE(nn.Module):
    """
    Choquet Adaline Operator
    :math:`P_{\\vec{\\omega}, \\vec{\\theta}}^{g,B,F} (x_1,...,x_n) = F(B_1(g_1(\\omega_1,x_1),\\theta_1),...,B_n(g_n(\\omega_n,x_n),\\theta_n))`.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.num_features = input_size
        self.weight = nn.Parameter(torch.zeros(input_size, output_size,
                                               dtype=torch.float), requires_grad=True)
        #  self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(input_size, output_size, dtype=torch.float), requires_grad=True)
        #  self.bias = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
        self.choquet = Choquet()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, 0,1)
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # for choquet f-mesasurens
        # self.weight.data.clamp_(min=0, max=1)
        # landa = bisectionLien(FI.FLambda, self.weight.t().detach().numpy()[0])
        # CFI = FI.ChoquetLambdaT(x, self.weight, landa)
        CFI = Choquet.apply(x,self.weight)
        # print("CFI: ",CFI)
        output = torch.sigmoid(CFI)
        return output

    def _sign(self, x):
        return 0 if x >= 0.5 else 1.
        return 1. if x >= 0 else -1.

    def predict(self, x):
        x = self.forward(x).detach()
        return x


def testCADALINE(datasets, run):
    for t in datasets:
        print(t)
        xd = []
        yd = []
        print("-- " + t[0][0] + " -- ")
        print("------ {}: Base -----".format("CADALINE"))
        plotname = t[0][0] + " - " + "CADALINE"
        x, y = t[0][1]()
        avg = []
        for i in range(1):#0):
            avgfold = []
            for train_index, test_index in Folds(t[0][0], i):
                print("prefolds",i)
                m, suc, total = bFolds(x, y, train_index, test_index, CADALINE)  # FIXME: CADALINE..
                print("post", i)
                avgfold.append((suc * 100 / total).tolist())
            avg.append(avgfold)
            print(t[0][0], "CADALINE", i, np.average(avgfold), avgfold)
            # print(avg)
            with open(os.path.join("..", "Results", t[0][0] + run + ".csv"), "a") as f:
                fwriter = csv.writer(f, delimiter=",")
                fwriter.writerow([t[0][0], i, "CADALINE", "", "", np.average(avgfold)] + avgfold)
            with open(os.path.join("..", "Results", run + ".csv"), "a") as f:
                fwriter = csv.writer(f, delimiter=",")
                fwriter.writerow([t[0][0], i, "CADALINE", "", "", np.average(avgfold)] + avgfold)
        print("{}: {} repetitions: {}% accuracy".format(t[0][0], i + 1, np.average(avg)))
        plt.show()
        xd.append("CADALINE")
        yd.append(np.average(avg))


if __name__=="__main__":
    ds = Datasets()

    run = "xx-CADALINE-20-06-2021"

    datasets = [[["Appendicitis", ds.Appendicitis, 85.8, 85.8]],
                [["Breast Cancer", ds.BreastCancer, 71.3, 75.3]],
                [["GermanCredit", ds.GermanCredit, 75.4, 73.1]],
                [["Breast Cancer W", ds.BreastCancerWisconsin, 96.1, 96.5]],
                [["Statlog HD", ds.StatLogHeart2, 84.5, 82.9]]]

    datasetsnoonline = [[["Diabetes", ds.Diabetes, 73.6, 76.8]],
                        [["Cleveland HD ", ds.KaggleClevelandHD, 83.5, 82.1]]]
    # dsthyroid = ["Thyroid",ds.Thyroid,97.4,96.2]
    datasets += datasetsnoonline

    testCADALINE(datasets,run)