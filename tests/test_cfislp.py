# -*- coding: utf-8 -*-

import unittest

import src.FIntegrals as FI
from src.pFISLP import FISLP


class TestUtils(unittest.TestCase):

    def test_FFISLP1(self):
        """
        Fitness function test
        mubinary = ['1100110011','1100110011','0111111111','1011001101','0111111111','0110011010']+['0111111011']
        mu = [0.8,0.8,0.5,0.7,0.5,0.4]+[0.495]
        """
        x = [[0.5, 0.5, 0.3, 0.4, 0.4, 0.4], [0.5, 0.5, 0.3, 0.4, 0.4, 0.4], [0.5, 0.5, 0.3, 0.4, 0.4, 0.4]]
        y = [0, 0, 1]
        F = FI.ChoquetLambda
        a = FISLP(0, 0, 0, 10, 0, 0, 1, 0.1, x, y, F)
        individual = '1100110011110011001101111111111011001101011111111101100110100111111011'  # + cut
        r = a.FFISLP(individual)
        expected = 2 - 0.1  # two correct and one incorrect
        self.assertAlmostEqual(r, expected, places=3)

    def test_FFISLP2(self):
        """
        Fitness function test
        mubinary = ['1100110011','1100110011','0111111111','1011001101','0111111111','0110011010']+['0111111011']
        mu = [0.8,0.8,0.5,0.7,0.5,0.4]+[0.495]
        """
        x = [[0.5, 0.5, 0.3, 0.4, 0.4, 0.4], [0.5, 0.5, 0.3, 0.4, 0.4, 0.4], [0.5, 0.5, 0.3, 0.4, 0.4, 0.4]]
        y = [0, 0, 0]
        F = FI.ChoquetLambda
        a = FISLP(0, 0, 0, 10, 0, 0, 1, 0.1, x, y, F)
        individual = '1100110011110011001101111111111011001101011111111101100110100111111011'  # + cut
        r = a.FFISLP(individual)
        expected = 3  # three correct
        self.assertAlmostEqual(r, expected, places=3)

    def test_FFISLP3(self):
        """
        Fitness function test
        mubinary = ['1100110011','1100110011','0111111111','1011001101','0111111111','0110011010']+['0111111011']
        mu = [0.8,0.8,0.5,0.7,0.5,0.4]+[0.495]
        """
        x = [[0.5, 0.5, 0.3, 0.4, 0.4, 0.4], [0.5, 0.5, 0.3, 0.4, 0.4, 0.4], [0.5, 0.5, 0.3, 0.4, 0.4, 0.4]]
        y = [1, 1, 1]
        F = FI.ChoquetLambda
        a = FISLP(0, 0, 0, 10, 0, 0, 1, 0.1, x, y, F)
        individual = '1100110011110011001101111111111011001101011111111101100110100111111011'  # + cut
        r = a.FFISLP(individual)
        expected = -0.1732  # Three incorrect
        self.assertAlmostEqual(r, expected, places=3)
    

if __name__ == '__main__':
    unittest.main()
