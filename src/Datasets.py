#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 13:36:07 2020

@author: asier_urio
"""
import os

import pandas as pd
from sklearn.preprocessing import minmax_scale

#https://fizyka.umk.pl/kis-old/projects/datasets.html
# + Appendicitis 106 - 7 numerical
# ?? Breast cancer 286 - 9 numerical ??
# + Wisconsin breast cancer - 699 - 9 numerical
# + Diabetes - 768 - 8 numerical
# ?? German credit 1000 - 24 numerical
# Cleveland heart disease 303 - 6 numerical - 7 categorical¿?
# Statlog heart disease 270 - 6 numerical - 7 categorical
# Thyroid 3772 -  6 numerical - 15 categorical (binary)

class Datasets():

    def __init__(self, path=None):
        if path:
            self.path = path
        else:
            self.path = "../Data"

    def GermanCredit(self, normalize=True):
        """

        Source: https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29

        Professor Dr. Hans Hofmann
        Institut f"ur Statistik und "Okonometrie
        Universit"at Hamburg
        FB Wirtschaftswissenschaften
        Von-Melle-Park 5
        2000 Hamburg 13

        This dataset requires use of a cost matrix (see below)

        . 1 2
        ----------------------------
        1 0 1
        -----------------------
        2 5 0

        (1 = Good, 2 = Bad)

        The rows represent the actual classification and the columns the predicted classification.

        It is worse to class a customer as good when they are bad (5), than it is to class a customer as bad when they are good (1).

        Parameters
        ----------
        normalize : TYPE, optional
            DESCRIPTION. The default is True.
            False for original data, not in [0,1] range

        Returns
        -------
        x : numpy array
            Attributes
        y : numpy array
            clases

        """
        datafile = os.path.join(self.path, "german.data")
        data = pd.read_csv(datafile, sep=" ", header=None)
        columns = [0, 2, 3, 4, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
        for c in columns:
            data = pd.concat([data, pd.get_dummies(data.iloc[:, c], prefix=c)], axis=1)
        data.drop(data.iloc[:, columns], axis=1, inplace=True)
        if normalize:
            data.iloc[:, :] = minmax_scale(data)
        y = data.iloc[:, 6].values
        x = data.drop(data.columns[[6]], axis=1).values

        return x, y

    def BreastCancerWisconsin(self, normalize=True):
        """

        Source: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

        Creators:

        1. Dr. William H. Wolberg, General Surgery Dept.
        University of Wisconsin, Clinical Sciences Center
        Madison, WI 53792
        wolberg '@' eagle.surgery.wisc.edu

        2. W. Nick Street, Computer Sciences Dept.
        University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
        street '@' cs.wisc.edu 608-262-6619

        3. Olvi L. Mangasarian, Computer Sciences Dept.
        University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
        olvi '@' cs.wisc.edu

        Donor:

        Nick Street

        Parameters
        ----------
        normalize : TYPE, optional
            DESCRIPTION. The default is True.
            False for original data, not in [0,1] range

        Returns
        -------
        x : numpy array
            Attributes
        y : numpy array
            clases

        """
        datafile = os.path.join(self.path, "breast-cancer-wisconsin.data")
        data = pd.read_csv(datafile, header=None)
        data.iloc[:, 6] = pd.to_numeric(data.iloc[:, 6], errors='coerce')
        data.dropna(inplace=True)
        data.iloc[:, 6] = data.iloc[:, 6].astype('int64')
        if normalize:
            data.iloc[:, :] = minmax_scale(data)
        else:
            #data.iloc[:, 1:-1] = minmax_scale(data.iloc[:, 1:-1])
            data.iloc[:, -1] = data.iloc[:, -1].apply(lambda x: 0 if x == 2 else 1)
        y = data.iloc[:, -1].values
        x = data.iloc[:, 1:-1].values

        return x, y

    def ProcessedClevelandHD(self, normalize=True):
        """

        Source: https://archive.ics.uci.edu/ml/datasets/Heart+Disease

        Creators:

        1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
        2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
        3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
        4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

        Donor:

        David W. Aha (aha '@' ics.uci.edu) (714) 856-8779

        Parameters
        ----------
        normalize : TYPE, optional
            DESCRIPTION. The default is True.
            False for original data, not in [0,1] range

        Returns
        -------
        x : numpy array
            Attributes
        y : numpy array
            clases

        """
        datafile = os.path.join(self.path, "processed.cleveland.data")
        data = pd.read_csv(datafile, header=None)
        data.iloc[:, 11] = pd.to_numeric(data.iloc[:, 11], errors='coerce')#?
        data.iloc[:, 12] = pd.to_numeric(data.iloc[:, 12], errors='coerce')#?
        data.dropna(inplace=True)#?
        data.iloc[:, -1] = data.iloc[:, -1].apply(lambda x: 0 if x == 0 else 1)
        if normalize:
            data.iloc[:, 1:-1] = minmax_scale(data.iloc[:, 1:-1])
        y = data.iloc[:, -1].values
        x = data.iloc[:, 1:-1].values

        return x, y

    def Diabetes(self, normalize=True):
        """

        Source: https://www.kaggle.com/uciml/pima-indians-diabetes-database

        Parameters
        ----------
        normalize : TYPE, optional
            DESCRIPTION. The default is True.
            False for original data, not in [0,1] range

        Returns
        -------
        x : numpy array
            Attributes
        y : numpy array
            clases

        """
        datafile = os.path.join(self.path, "diabetes.csv")
        data = pd.read_csv(datafile)
        if normalize:
            data.iloc[:, :] = minmax_scale(data)
        y = data.iloc[:, -1].values
        x = data.iloc[:, :-1].values

        return x, y

    def StatlogHeart(self, normalize=True):
        """

        Source: https://www.openml.org/d/53

        Parameters
        ----------
        normalize : TYPE, optional
            DESCRIPTION. The default is True.
            False for original data, not in [0,1] range

        Returns
        -------
        x : numpy array
            Attributes
        y : numpy array
            clases

        """
        #FIXME: unfinished
        datafile = os.path.join(self.path, "dataset_53_heart-statlog.data")
        data = pd.read_csv(datafile, header=None)
        columns = [2, 6]#binary=.[1, 5, 8]
        for c in columns:
            data = pd.concat([data, pd.get_dummies(data.iloc[:, c], prefix=c)], axis=1)
        data.drop(data.iloc[:, columns], axis=1, inplace=True)
        if normalize:
            data.iloc[:, :] = minmax_scale(data)
        y = data.iloc[:, -1].values #!!
        x = data.iloc[:, :-1].values

        return x, y

    def Appendicitis(self, normalize=True):
        """
        Source: https://sci2s.ugr.es/keel/dataset.php?cod=183

        This dataset was proposed in S. M. Weiss, and C. A. Kulikowski, Computer Systems That Learn (1991).

        The data represents 7 medical measures taken over 106 patients on which the class label represents if the patient has appendicitis (class label 1) or not (class label 0).

        Parameters
        ----------
        normalize : TYPE, optional
            DESCRIPTION. The default is True.
            False for original data, not in [0,1] range

        Returns
        -------
        x : numpy array
            Attributes
        y : numpy array
            clases

        """
        datafile = os.path.join(self.path, "appendicitis.dat")
        data = pd.read_csv(datafile, skiprows=12, header=None)
        if normalize:
            data.iloc[:, 1:-1] = minmax_scale(data.iloc[:, 1:-1])
        x = data.iloc[:, 1:-1].values
        y = data.iloc[:, -1].values

        return x, y

    def Thyroid(self, normalize=True):
        """

        Source: https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease
        Ross Quinlan


        Parameters
        ----------
        normalize : TYPE, optional
            DESCRIPTION. The default is True.
            False for original data, not in [0,1] range

        Returns
        -------
        x : numpy array
            Attributes
        y : numpy array
            clases

        """
        datafile = os.path.join(self.path, "ann-train.data")
        data = pd.read_csv(datafile, sep=" ", header=None)
        data = data.iloc[:, 0:22]
        data.iloc[:, -1] = data.iloc[:, -1].apply(lambda x: 0 if x == 3 else 1)
        #columns = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]#binary data
        #for c in columns:
            #data = pd.concat([data,pd.get_dummies(data.iloc[:,c], prefix=c)],axis=1)
        #data.drop(data.iloc[:,columns],axis=1, inplace=True)
        if normalize:
            data.iloc[:, :] = minmax_scale(data)
        y = data.iloc[:, -1].values
        x = data.iloc[:, :-1].values

        return x, y

    def BreastCancer(self, normalize=True):
        """

        Source: https://archive.ics.uci.edu/ml/datasets/breast+cancer
        Creators:

        Matjaz Zwitter & Milan Soklic (physicians)
        Institute of Oncology
        University Medical Center
        Ljubljana, Yugoslavia

        Donors:

        Ming Tan and Jeff Schlimmer (Jeffrey.Schlimmer '@' a.gp.cs.cmu.edu)


        Parameters
        ----------
        normalize : TYPE, optional
            DESCRIPTION. The default is True.
            False for original data, not in [0,1] range

        Returns
        -------
        x : numpy array
            Attributes
        y : numpy array
            clases

        """
        datafile = os.path.join(self.path, "breast-cancer.data")
        data = pd.read_csv(datafile, header=None)
        data.iloc[:, 0] = data.iloc[:, 0].apply(lambda x: 0 if x == "no-recurrence-events" else 1)
        columns = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        for c in columns:
            data = pd.concat([data, pd.get_dummies(data.iloc[:, c], prefix=c)], axis=1)
        data.drop(data.iloc[:, columns], axis=1, inplace=True)
        if normalize:
            data.iloc[:, :] = minmax_scale(data)
        y = data.iloc[:, 0].values
        x = data.iloc[:, 1:].values

        return x, y
