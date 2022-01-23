#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 12:02:33 2020

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

# from FIntegrals import ChoquetCardinal

'''
https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
unesqueeze añade una dimensión: [a] -> [[a]]
https://pytorch.org/docs/stable/generated/torch.squeeze.html
squeeze elimina una dimensión: [[a]] -> [a]

https://pytorch.org/docs/stable/generated/torch.mean.html
If keepdim is True, the output tensor is of the same size as input except 
in the dimension(s) dim where it is of size 1. Otherwise, dim is squeezed 
(see torch.squeeze()), resulting in the output tensor having 1 (or len(dim)) 
fewer dimension(s).

'''

"""
Function definitions for the generalized adaline operator
the following functions need to be adjusted to work along the axis
max = torch.max
min = torch.min
avg = torch.mean
sum = torch.sum
Move this functions to other file?
"""


# Functions for generating the ouput (on for layer)
# n: output layers
# x: input cases
# d: input cases dimensions
# input: torch.Tensor (x * d)
# output: torch.Tensor (1 * n)


# The following functions operate on the highest dimension of a tensor.
# Tested
def vmean(x):
    return torch.mean(x, dim=x.dim() - 1, keepdim=x.dim() < 3).t()  # keepdim not necessary, default False


# Tested
def vsum(x):
    return torch.sum(x, dim=x.dim() - 1, keepdim=x.dim() < 3).t()  # keepdim not necessary, default False


# Tested
def vmax(x):
    return torch.max(x, dim=x.dim() - 1, keepdim=x.dim() < 3).values.t()


# Tested
def vmin(x):
    return torch.min(x, dim=x.dim() - 1, keepdim=x.dim() < 3).values.t()


# UnTested
def vprod(x):
    return torch.prod(x, dim=x.dim() - 1, keepdim=x.dim() < 3).t()  # keepdim not necessary, default False


# Functions for inputs and weights or bias

# Tested
def dot(y, w):
    """
       producto de input por pesos
       salida esperada: lista con una lista por neurona y dentro una por caso de input
       una neurona 1 caso: [[[caso]]]
       dos neuronas 1 caso: [[[caso]],[[caso2]]]
       una neurona x casos:[[[caso],[caso]]]
       dos neurona x casos:[[[caso],[caso]],[[caso],[caso]]]
    """
    wdim1, _ = w.size()
    return y.expand([wdim1, -1, -1]) * torch.unsqueeze(w, 1)


def div(y, w):
    """
       producto de input por pesos
       salida esperada: lista con una lista por neurona y dentro una por caso de input
       una neurona 1 caso: [[[caso]]]
       dos neuronas 1 caso: [[[caso]],[[caso2]]]
       una neurona x casos:[[[caso],[caso]]]
       dos neurona x casos:[[[caso],[caso]],[[caso],[caso]]]
    """
    wdim1, _ = w.size()
    # TODO: to avoid / 0 -> w + epsilon?
    return torch.true_divide(y.expand([wdim1, -1, -1]), torch.unsqueeze(w, 1))


def madd(y, w):
    """
       producto de input por pesos
       salida esperada: lista con una lista por neurona y dentro una por caso de input
       una neurona 1 caso: [[[caso]]]
       dos neuronas 1 caso: [[[caso]],[[caso2]]]
       una neurona x casos:[[[caso],[caso]]]
       dos neurona x casos:[[[caso],[caso]],[[caso],[caso]]]
    """
    wdim1, _ = w.size()
    return y.expand([wdim1, -1, -1]) + torch.unsqueeze(w, 1)


def mminus(y, w):
    """
       producto de input por pesos
       salida esperada: lista con una lista por neurona y dentro una por caso de input
       una neurona 1 caso: [[[caso]]]
       dos neuronas 1 caso: [[[caso]],[[caso2]]]
       una neurona x casos:[[[caso],[caso]]]
       dos neurona x casos:[[[caso],[caso]],[[caso],[caso]]]
    """
    wdim1, _ = w.size()
    return y.expand([wdim1, -1, -1]) - torch.unsqueeze(w, 1)


# I supose the following will work, as the use operators that work for all elements
# disregardin shape and previously defined functions (dot,div)
def absdot(x, o):
    return dot(torch.abs(o), x)


def sqrdot(x, o):
    return dot((o ** 2), x)


def cavg(y, w):
    return torch.true_divide(madd(y, w), 2)


def sgnavg(y, w):
    return torch.sign(mdot(y, w)) * torch.true_divide(madd(torch.abs(y),torch.abs(w)), 2)


def sgnmax(y, w):
    return mdot(torch.sign(mdot(y, w)), vmax(torch.abs(y), torch.abs(w)))  # FIXME: vmax not working here a dot like function needed


def sgnmin(y, w):
    return torch.sign(mdot(y, w)) * torch.min(torch.abs(y), torch.abs(w))  # FIXME: vmax not working here a dot like function needed


"""
In the next functions there is a call to view to avoid
the following warning:
    loss.py:516: UserWarning: Using a target size (torch.Size([105, 1])) 
    that is different to the input size (torch.Size([105])) is deprecated.
"""


def hamacher(x, y, epsilon=1e-10):
    print("hx", x)
    print("hy", y)
    return x * y / (epsilon + x + y - x * y)  # epsilon to avoid 0/0 nan
    # For multiple layers


# adaptadas a multilayer


# B's
def madd2(x, y):
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
F = [vsum]
B = [torch.add]
g = [mdot, m2min, mavg, t_luckas, mgeom, oxy, m2max, probsum, c_luckas]  # mhamacher
