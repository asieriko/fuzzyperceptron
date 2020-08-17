#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:04:07 2020

@author: asier
"""
import torch
import numpy as np

from utils import bisectionLien, generateGMeasure

class FIntegrals():
    """
     Class containing several Fuzzy Integrals
    
    """
    
    def ChoquetTorch(self,x,mu):
        x = torch.tensor(x)
        w = torch.rand(2**x.size()[0])
        mu = torch.tensor(mu)
        mudic = {'0':0}
        def genmudic(mu,n):
            e1 = mu[0]
            for el in mu[1:]:
                pass
            
        w[0]=0
        w[-1]=1
        xsorted = x.sort()
        #values=tensor([1, 2, 4, 5, 9]),
        #indices=tensor([0, 2, 1, 4, 3]))
        print(xsorted)
        print(xsorted.values)
        print(xsorted.indices)
        print(w)
        choquet = 0
        for i in range(len(x)):
            print("Measure",xsorted.indices[i:].sort().values)
            choquet += xsorted.values[i]*w[i]
            
    def ChoquetLambda(self,x,mu,l=None,verbose=False):
        """Choquet Lambda integral
        
        :param x: values 
        :param mu: fuzzy measure
        
        :rtype: Choquet lambda-Integral with lambda computed based on :math:`\\mu`
    
        
        :math:`CFI(x)_{\\mu} = \\displaystyle\\sum_{i=1}^{n}(f(x_{\\sigma_i})-f(x_{\\sigma_{(i-1)}}))* g(A_j)`.
        
        """
        
        x = np.array(x)
        mu = np.array(mu)
        
        sortedindex = np.argsort(x)
        xsorted = np.take(x, sortedindex)
        musorted = np.take(mu, sortedindex)
              
        if l == None:
            l = bisectionLien(self.FLambda,mu)
        
        nmu = generateGMeasure(musorted,l)

        choquet = sum(np.diff(np.append([0],xsorted)) * nmu)
        #For numpy >= 1.16
        #choquet = sum(np.diff(xsorted,prepend=[0]) * nmu)

        if verbose:
            print("X:",x,"X sorted:", xsorted)
            print("mu",mu,"Mu sorted:", musorted)
            print("Lambda:",l)
            print("Generated measures:",nmu)
            print("Choquet:",choquet)
    
        return choquet

    def GeneralizedChoquetLambda(self,x,mu,F1,F2,l=None,verbose=False):
        """Choquet Lambda integral
        
        :param x: values 
        :param mu: fuzzy measure
        :param F1: Function1
        :param F2: Function2
        
        :rtype: Generalized Choquet lambda-Integral with lambda computed based on :math:`\\mu`
    
        
        :math:`CFI(x)_{\\mu} = F2_{i=1}^{n}F1(f(x_{\\sigma_i})-f(x_{\\sigma_{(i-1)}})),g(A_j))`.
        
        """
        
        x = np.array(x)
        mu = np.array(mu)
        
        sortedindex = np.argsort(x)
        xsorted = np.take(x, sortedindex)
        musorted = np.take(mu, sortedindex)
                
        if l == None:
            l = bisectionLien(self.FLambda,mu)
        nmu = generateGMeasure(musorted,l)

        gchoquet = F2(F1(np.diff(np.append([0],xsorted)),nmu))
        #For numpy >= 1.16
        #gchoquet = F2(F1(np.diff(xsorted,prepend=[0]),nmu))

        if verbose:
            print("X:",x,"X sorted:", xsorted)
            print("mu",mu,"Mu sorted:", musorted)
            print("Lambda:",l)
            print("Generated measures:",nmu)
            print("Choquet:",gchoquet)
    
        return gchoquet


    def SugenoLambda(self,x,mu,l=None,verbose=False):
        """Sugeno Lambda integral
        
        :param x: values 
        :param mu: fuzzy measure
        
        :rtype: Sugeno lambda-Integral with lambda computed based on :math:`\\mu`
        
        :math:`SFI(x)_{\\mu} = \\displaystyle\\wedge_{i=1}^{n}\\vee(f(x_{\\sigma_i}),g(A_j))`.
        
        """
        
        x = np.array(x)
        mu = np.array(mu)
        
        sortedindex = np.argsort(x)
        xsorted = np.take(x, sortedindex)
        musorted = np.take(mu, sortedindex)
                
        if l == None:
            l = bisectionLien(self.FLambda,mu)
        nmu = generateGMeasure(musorted,l)

        sugeno = max([min(a,b) for a,b in zip(xsorted,nmu)])

        if verbose:
            print("X:",x,"X sorted:", xsorted)
            print("mu",mu,"Mu sorted:", musorted)
            print("Lambda:",l)
            print("Generated measures:",nmu)
            print("Sugeno:",sugeno)
    
        return sugeno              
        
    
    def GeneralizedSugenoLambda(self,x,mu,F1,F2,l=None,verbose=False):
        """Generalized Sugeno Lambda integral
        
        :param x: values 
        :param mu: fuzzy measure
        
        :rtype: Generalized Sugeno lambda-Integral with lambda computed based on :math:`\\mu`
        
        :math:`SFI(x)_{\\mu} = F2_{i=1}^{n}F1(f(x_{\\sigma_i}),g(A_j))`.
        
        """
        
        x = np.array(x)
        mu = np.array(mu)
        
        sortedindex = np.argsort(x)
        xsorted = np.take(x, sortedindex)
        musorted = np.take(mu, sortedindex)
                
        if l == None:
            l = bisectionLien(self.FLambda,mu)
        nmu = generateGMeasure(musorted,l)

        gsugeno = F2([F1(a,b) for a,b in zip(xsorted,nmu)])

        if verbose:
            print("X:",x,"X sorted:", xsorted)
            print("mu",mu,"Mu sorted:", musorted)
            print("Lambda:",l)
            print("Generated measures:",nmu)
            print("Sugeno:",gsugeno)
    
        return gsugeno          
    
    def FLambda(self,l,mu):
        """
        :param l: lambda
        :param mu: fuzzy measure (numpy array)
        
        :rtype: F(l)
        
        :math:`F(\\lambda) = \\displaystyle\\prod_{j=1}^{n}(1+\\lambda * \\mu_{j}) - \\lambda -1`."""
        #print(l,mu,np.product((mu*l)+1.0)-l-1)
        return np.product((mu*l)+1.0)-l-1
        #f = 1
        #for m in mu:
        #    f *= (1+m*l)
        #f += -l-1
        #return f
    
    def DFlambda(self,l,mu):
        """:math:`F'(\\lambda) = \\sigma_{i}\\mu_{i}\\displaystyle\\prod_{j=1 not i}^{n}(1+\\lambda * \\mu_{j}) - 1`."""
        df = -1
        for i in range(len(mu)):
            p = mu[i]
            for j in range(len(mu)): 
                if i != j:
                    p *= (1+mu[j]*l)
            df += p
        return df


def FLambda(l,mu):
        """:math:`F(\\lambda) = \\displaystyle\\prod_{j=1}^{n}(1+\\lambda * \\mu_{j}) - \\lambda -1`."""
        f = 1
        for m in mu:
            f *= (1+m*l)
        f += -l-1
        return f
    
def F2(l,mu):
    return np.product((mu*l)+1.0)-1-l
    
if __name__ == "__main__":
    FI = FIntegrals()
    mu = [0.624801,0.029406,0.326642,0.374676]
    x = [6,3,9,2]
    FI.ChoquetLambda(x,mu,True)
    