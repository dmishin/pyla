import unittest
from pyla.core import *
from pyla.nmead import *
import math

from pyla_test_case import PylaTestCase


def rosenbrock(z):
    return (1-z[0])**2+100.0*(z[1]-z[0]**2)**2

class TestnelderMeadOptimizer(PylaTestCase):

    def test_rozenbrock2d(self):
        xbest, fbest, steps, calculations \
            = nmead([[.1,0.],[0.,.1],[-.1,0.]],
                    rosenbrock,
                    tol = 1e-6)
        self.assertVectorEqual( xbest, [1.0, 1.0] )

