#  -*- coding: utf-8 -*-

import unittest

from src import utils as utils


class TestUtils(unittest.TestCase):
    
    def test_bisectionLien1(self):
        # Chung-Chang Lien & Chie-Bein Chen
        mu = [0.624801,0.029406,0.326642,0.374676]
        ll = utils.bisectionLien(utils.F,mu)
        espl=-0.6592807
        self.assertAlmostEqual(ll, espl, places=5)

    def test_bisectionLien2(self):
        # Chung-Chang Lien & Chie-Bein Chen
        mu = [0.999702,0.844691,0.724534,0.694047]
        ll = utils.bisectionLien(utils.F,mu)
        espl=-0.9999956
        self.assertAlmostEqual(ll, espl, places=5)

    def test_bisectionLien3(self):
        # C.G. Magadun an M.S. Bapat
        # SCILAB resolving 6th grade equation
        # http://www.researchmathsci.org/IJFMAart/IJFMA-v15n2-1.pdf
        mu = [0.8,0.8,0.5,0.7,0.5,0.4]
        ll = utils.bisectionLien(utils.F,mu)
        espl=-0.9981565
        self.assertAlmostEqual(ll, espl, places=5)


    def test_bisectionLien4(self):
        mu = [0.3,0.2,0.4,0.1]
        ll = utils.bisectionLien(utils.F,mu)
        espl= 0
        self.assertAlmostEqual(ll, espl, places=5)


    def test_bisectionLien5(self):
        # find esp lambda
        mu = [0.3,0.2,0.25,0.1]
        ll = utils.bisectionLien(utils.F,mu)
        espl= 0.538635
        self.assertAlmostEqual(ll, espl, places=5)

    def test_bisectionLien6(self):
        mu = [0.2,0.3]
        ll = utils.bisectionLien(utils.F,mu)
        espl= 8.333332
        self.assertAlmostEqual(ll, espl, places=5)

    def test_bisectionWang1(self):
        # Chung-Chang Lien & Chie-Bein Chen
        mu = [0.624801,0.029406,0.326642,0.374676]
        lw = utils.bisectionWang(utils.F,utils.Fprime,mu)
        espw=-0.659279
        self.assertAlmostEqual(lw, espw, places=5)
        
    def test_bisectionWang2(self):
        # Chung-Chang Lien & Chie-Bein Chen
        mu = [0.999702,0.844691,0.724534,0.694047]
        lw = utils.bisectionWang(utils.F,utils.Fprime,mu)
        espw=-0.999995
        self.assertAlmostEqual(lw, espw, places=5)

    def test_bisectionWang3(self):
        # C.G. Magadun an M.S. Bapat
        # SCILAB resolving 6th grade equation
        # http://www.researchmathsci.org/IJFMAart/IJFMA-v15n2-1.pdf
        mu = [0.8,0.8,0.5,0.7,0.5,0.4]
        lw = utils.bisectionWang(utils.F,utils.Fprime,mu)
        espw=-0.9981565
        self.assertAlmostEqual(lw, espw, places=5)


    def test_bisectionWang4(self):
        mu = [0.3,0.2,0.4,0.1]
        ll = utils.bisectionWang(utils.F, utils.Fprime,mu)
        espl= 0
        self.assertAlmostEqual(ll, espl, places=5)

    def test_bisectionWang5(self):
        # find esp lambda
        mu = [0.3,0.2,0.25,0.1]
        ll = utils.bisectionWang(utils.F, utils.Fprime,mu)
        espl= 0.538635
        self.assertAlmostEqual(ll, espl, places=5)

    def test_bisectionWang6(self):
        mu = [0.2,0.3]
        ll = utils.bisectionWang(utils.F, utils.Fprime,mu)
        espl= 8.333332
        self.assertAlmostEqual(ll, espl, places=5)


    def test_F_mu1(self):
        mu = [0.5,0.4,0.1]
        l = utils.F(0,mu)
        espl= 0
        self.assertEqual(l,espl)
        
    def test_GenerateMuLambda1(self):
        # C.G. Magadun an M.S. Bapat
        # http://www.researchmathsci.org/IJFMAart/IJFMA-v15n2-1.pdf
        mu = [0.4,0.5,0.5,0.7,0.8,0.8]
        l = -0.9981565
        expected = [1.0, 0.9987725, 0.9957093, 0.9895943, 0.9611798, 0.8]
        out = list(utils.generateGMeasure(mu, l))
        for e,o in zip(expected,out):
            self.assertAlmostEqual(o, e, places=3)

    def test_GenerateMuLambda2(self):
        # C.G. Magadun an M.S. Bapat
        # http://www.researchmathsci.org/IJFMAart/IJFMA-v15n2-1.pdf
        mu = [0.8,0.8,0.5,0.7,0.5,0.4]
        l = -0.9981565
        expected = [1.0, 0.992679806032971, 0.9563469279183847, 0.911014400788915, 0.7003687, 0.4]
        out = list(utils.generateGMeasure(mu, l))
        for e,o in zip(expected,out):
            self.assertAlmostEqual(o, e, places=3)
            
    def test_GenerateMuLambda3(self):
        # C.G. Magadun an M.S. Bapat
        # http://www.researchmathsci.org/IJFMAart/IJFMA-v15n2-1.pdf
        # FChoquet fails with this data. An error in the paper?
        mu = [0.7,0.5,0.5,0.8,0.8,0.4]
        l = -0.9981565
        expected = [1.0, 0.9957168, 0.9896093, 0.9774167, 0.8805899, 0.4]
        out = list(utils.generateGMeasure(mu, l))
        for e,o in zip(expected,out):
            self.assertAlmostEqual(o, e, places=3)     
            
               
# Testing bintofloat makes no necesary individualtofloat testing for precision...
    
    def test_IndividualtoFloat0(self):
        x = [0,0,0,0,0,0,0,0,0,0]
        r = utils.individualtofloat(x)
        expected = [0.0]
        self.assertAlmostEqual(r, expected,places=3)

    def test_IndividualtoFloat1(self):
        x = [1,1,1,1,1,1,1,1,1,1]
        r = utils.individualtofloat(x)
        expected = [1.0]
        self.assertAlmostEqual(r, expected,places=3)

    def test_IndividualtoFloat5(self):
        x = [1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        r = utils.individualtofloat(x)
        expected = [1.0,0.5,1.0]
        for e,o in zip(expected,r):
            self.assertAlmostEqual(o, e, places=3)

    def test_IndividualtoFloat6(self):
        x = [1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        with self.assertRaises(ValueError) as e:
            utils.individualtofloat(x)    
            
    def test_BintoFloat0(self):
        x = [0,0,0,0,0,0,0,0,0,0]
        r = utils.bintofloat(x)
        expected = 0.0
        self.assertAlmostEqual(r, expected,places=3)

    def test_BintoFloat1(self):
        x = [1,1,1,1,1,1,1,1,1,1]
        r = utils.bintofloat(x)
        expected = 1.0
        self.assertAlmostEqual(r, expected,places=3)            

    def test_BintoFloat2(self):
        x = [1,0,0,0,0,0,0,0,0,0]
        r = utils.bintofloat(x)
        expected = 0.5
        self.assertAlmostEqual(r, expected,places=3)      

    def test_BintoFloat3(self):
        x = [0,1,1,1,1,1,1,1,1,1]
        r = utils.bintofloat(x)
        expected = 0.5
        self.assertAlmostEqual(r, expected,places=3)  
        
                  
if __name__ == '__main__':
    unittest.main()        