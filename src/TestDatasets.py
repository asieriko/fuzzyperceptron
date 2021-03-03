#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 23:37:33 2021

@author: asier_urio

Simple program to perform test over the dataset
It's important to use the Datasets file and the
GADALINE one
Also a TensorFlow implementation is provided to
compare results
"""
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from Datasets import Datasets, Folds
from GADALINE import bFolds, SLP, MLP, trainTest

tf.get_logger().setLevel("FATAL")


def tensorFlow(x, y, train_index, test_index, m="SLP"):
    # A simple tensorFlow implementation to
    # compare results with pytorch

    # x_train, x_test = x[train_index], x[test_index]
    # y_train, y_test = y[train_index], y[test_index]

    x_train, x_test = tf.constant(x[train_index]), tf.constant(x[test_index])
    y_train, y_test = tf.constant(y[train_index]), tf.constant(y[test_index])

    # define the keras model
    model = Sequential()
    if m == "SLP":
        model.add(Dense(1, input_dim=len(x[0]), activation='sigmoid'))
    else:  # MLP
        model.add(Dense(units=3, input_dim=len(x[0]), activation='sigmoid'))
        model.add(Dense(units=1, input_dim=3, activation='sigmoid'))
    # compile the keras model
    sgd = tf.keras.optimizers.SGD(learning_rate=0.9, momentum=0.5, name="SGD")
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=sgd, metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(x_train, y_train, epochs=10000, batch_size=len(x[0]), verbose=0)
    scores = model.evaluate(x_test, y_test, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    return scores[1] * len(y), len(y)


def testDatasetsSLP(datasets):
    for model in [SLP, MLP]:
        for t in datasets:
            # Thyroid mix all data an 10-folds
            # if t[0] == "Thyroid":
            #   x_train, y_train, x_test, y_test = ds.Thyroid()
            #   x = np.concatenate((x_train, x_test))
            #   y = np.concatenate((y_train, y_test))
            print("{} Results for {} dataset:".format(model, t[0]), end=" ")
            x, y = t[1]()
            avg = []
            for i in range(10):
                for train_index, test_index in Folds(t[0],i):
                    m, suc, tot = bFolds(x, y, train_index, test_index, model)
                    avg.append(suc / tot)
            print("Accuracy: {:.2f}".format(np.average(avg)))

        # Thyroid train vs test data
        print("{} Results for Thyroid dataset:".format(model), end=" ")
        x_train, y_train, x_test, y_test = ds.Thyroid()
        m, stats = trainTest(x_train, y_train, x_test, y_test, model)
        print("Accuracy: {:.2f}".format(stats.average(0)))


def testDatasetsPTTF(datasets,repetitions=10):
    dr = {}
    for mod in ["SLP", "MLP"]:
        print("Test", mod)
        print("Dataset\t Expected\t PyTorch\t TensorFlow")
        drp = {}
        m, er = (SLP, 2) if mod == "SLP" else (MLP, 3)
        for t in datasets:
            x, y = t[1]()
            avgpt = []
            avgtf = []
            for i in range(repetitions):
                avgptR = []
                avgtfR = []
                for train_index, test_index in Folds(t[0],i):
                    x_train, x_test = x[train_index], x[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    # _, sucpt, totpt = bFolds(x_train, y_train, x_test, y_test, m)
                    _, sucpt, totpt = bFolds(x, y, train_index, test_index, m)
                    acpt = sucpt / totpt  # FIXME: How to output data
                    print("*", end="")  # To see some progress...
                    suctf, tottf = tensorFlow(x, y, train_index, test_index, m=mod)
                    # suctf, tottf = tensorFlow(x_train, y_train, x_test, y_test, m=mod)
                    print("x", end="")  # To see some progress...
                    actf = suctf / tottf
                    avgpt.append(acpt*100)
                    avgtf.append(actf*100)
                avgptR.append(avgpt)
                avgtfR.append(avgtf)
                print(end="\r{}:".format(i))
                print("\rRep: {} {:.2f}% {:.2f}%".format(i + 1, np.average(avgptR), np.average(avgtfR)))
            drp[t[0]] = [t[0], t[er], np.average(avgpt), np.average(avgtf)]
            print("{}:\t {}%\t {:.2f}%\t {:.2f}%".format(t[0], t[er], np.average(avgpt), np.average(avgtf)))
        print(drp)
        dr[mod] = drp
    return dr


if __name__=="__main__":

    ds = Datasets()

    datasets = [["GermanCredit", ds.GermanCredit, 75.4, 73.1],
                ["Breast Cancer W", ds.BreastCancerWisconsin, 96.1, 96.5],
                ["Breast Cancer", ds.BreastCancer, 71.3, 75.3],
                ["Statlog HD", ds.StatLogHeart2, 84.5, 82.9],
                ["Appendicitis", ds.Appendicitis, 85.8, 85.8],
                ["Thyroid", ds.ThyroidTrain, 97.4, 96.2]]

    datasetsnoonline = [["Diabetes", ds.Diabetes, 73.6, 76.8],
                        ["Cleveland HD ", ds.KaggleClevelandHD, 83.5, 82.1]]

    datasets += datasetsnoonline

    resultdic = testDatasetsPTTF(datasets,10)
    print("Dataset\t Expected\t PyTorch\t TensorFlow")
    for k in resultdic.keys():
        print(k)
        for v in resultdic[k].values():
            print("{}\t {:.2f}\t {:.2f}\t {:.2f}\t".format(v[0], v[1], v[2], v[3]))
