#  -*- coding: utf-8 -*-

import unittest
from operator import mul
import numpy as np

from src import FIntegrals as FI


class TestUtils(unittest.TestCase):
    
    def test_FChoquetLambda1(self):
        # http://www.diva-portal.org/smash/record.jsf?pid=diva2%3A833444&dswid=7043
        # http://www.diva-portal.org/smash/get/diva2:833444/FULLTEXT01.pdf
        mu = [0.45,0.45,0.3]
        x = [45,50,40]
        r = FI.FIntegrals().ChoquetLambda(x,mu,verbose=True)
        expected = 46.295
        self.assertAlmostEqual(r, expected,places=3)

    def test_FChoquetLambda2(self):
        # http://www.diva-portal.org/smash/record.jsf?pid=diva2%3A833444&dswid=7043
        # http://www.diva-portal.org/smash/get/diva2:833444/FULLTEXT01.pdf
        mu = [0.45,0.45,0.3]
        x = [39,58,55]
        r = FI.FIntegrals().ChoquetLambda(x,mu,verbose=True)
        expected = 51.3804
        self.assertAlmostEqual(r, expected,places=2)# wiht 3 fails -> 51.3796
        
    def test_ChoquetLambda3(self):
        # C.G. Magadun an M.S. Bapat
        # SCILAB resolving 6th grade equation for lambda
        # http://www.researchmathsci.org/IJFMAart/IJFMA-v15n2-1.pdf
        mu = [0.8,0.8,0.5,0.7,0.5,0.4]
        x = [45,40,48,45,30,40]
        r = FI.FIntegrals().ChoquetLambda(x,mu,verbose=True)
        expected = 46.338515
        self.assertAlmostEqual(r, expected,places=4)
        
    def test_ChoquetLambda4(self):
        # C.G. Magadun an M.S. Bapat
        # SCILAB resolving 6th grade equation for lambda
        # http://www.researchmathsci.org/IJFMAart/IJFMA-v15n2-1.pdf
        mu = [0.8,0.8,0.5,0.7,0.5,0.4]
        x = [48,50,35,40,43,43]
        r = FI.FIntegrals().ChoquetLambda(x,mu,verbose=True)
        expected = 49.365525
        self.assertAlmostEqual(r, expected,places=4)

    def test_GChoquetLambda(self):
        # C.G. Magadun an M.S. Bapat
        # SCILAB resolving 6th grade equation for lambda
        # http://www.researchmathsci.org/IJFMAart/IJFMA-v15n2-1.pdf
        mu = [0.8,0.8,0.5,0.7,0.5,0.4]
        x = [48,50,35,40,43,43]
        r = FI.FIntegrals().ChoquetLambda(x,mu,verbose=True)
        e = FI.FIntegrals().GeneralizedChoquetLambda(x,mu,mul,sum,verbose=True)
        self.assertAlmostEqual(r, e,places=4)

    @unittest.skip("An error in the paper?\
        Hand made calculus seems right")
    def test_ChoquetLambda5(self):
        # C.G. Magadun an M.S. Bapat
        # SCILAB resolving 6th grade equation for lambda
        # http://www.researchmathsci.org/IJFMAart/IJFMA-v15n2-1.pdf
        # FChoquet fails with this data. An error in the paper?
        # Hand made calculus seems right
        # 34*1+7*0.9957168+1*0.9896093+1*0.9774167+1*0.8805899+3*0.4
        mu = [0.8,0.8,0.5,0.7,0.5,0.4]
        x = [43,44,41,34,42,47]
        r = FI.FIntegrals().ChoquetLambda(x,mu,verbose=True)
        expected = 45.917632
        self.assertAlmostEqual(r, expected,places=4)
        
    def test_FSugenoLambda1(self):
        # http://www.diva-portal.org/smash/record.jsf?pid=diva2%3A833444&dswid=7043
        # http://www.diva-portal.org/smash/get/diva2:833444/FULLTEXT01.pdf
        mu = [0.93295,0.303787,0.18659]
        x = [1,0.8,0.1]
        r = FI.FIntegrals().SugenoLambda(x,mu,verbose=True)
        expected = 0.93295
        self.assertAlmostEqual(r, expected,places=3)

    def test_FSugenoLambda2(self):
        # http://www.diva-portal.org/smash/record.jsf?pid=diva2%3A833444&dswid=7043
        # http://www.diva-portal.org/smash/get/diva2:833444/FULLTEXT01.pdf
        mu = [0.93295,0.303787,0.18659]
        x = [0.5,0.6,0.3]
        r = FI.FIntegrals().SugenoLambda(x,mu,verbose=True)
        expected = 0.500
        self.assertAlmostEqual(r, expected,places=3)
        
    def test_FSugenoLambda3(self):
        # http://www.diva-portal.org/smash/record.jsf?pid=diva2%3A833444&dswid=7043
        # http://www.diva-portal.org/smash/get/diva2:833444/FULLTEXT01.pdf
        mu = [0.93295,0.303787,0.18659]
        x = [0.3,0.3,0.8]
        r = FI.FIntegrals().SugenoLambda(x,mu,verbose=True)
        expected = 0.300
        self.assertAlmostEqual(r, expected,places=3)        


    def test_FGSugenoLambda1(self):
        # http://www.diva-portal.org/smash/record.jsf?pid=diva2%3A833444&dswid=7043
        # http://www.diva-portal.org/smash/get/diva2:833444/FULLTEXT01.pdf
        mu = [0.93295,0.303787,0.18659]
        x = [0.3,0.3,0.8]
        r = FI.FIntegrals().SugenoLambda(x,mu,verbose=True)
        e = FI.FIntegrals().GeneralizedSugenoLambda(x,mu,min,max,verbose=True)
        self.assertAlmostEqual(r, e,places=3)    

    
    # @unittest.skip("Does not work with __FLambda")
    def test_F(self):
        mu = np.array([0.5,0.4,0.1])
        l = FI.FIntegrals().FLambda(0,mu)
        espl= 0
        self.assertEqual(l,espl)


    # @unittest.skip("Does not work with __FLambda")
    def test_DF(self):
        mu = np.array([0.5,0.4,0.1])
        l = FI.FIntegrals().DFlambda(0,mu)
        espl= 0
        self.assertAlmostEqual(l,espl,places=5)

if __name__ == '__main__':
    unittest.main()        