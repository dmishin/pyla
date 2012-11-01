import unittest
from pyla.core import *
from pyla.nmead import *
import math

from pyla_test_case import PylaTestCase


def rozenbrock(z):
    return (1-z[0])**2+100.0*(z[1]-z[0]**2)**2

class TestnelderMeadOptimizer(PylaTestCase):

    def test_simple(self):
        def func(xyz):
            x,y,z = xyz
            return x**2 + y**2 + z**2

        xbest, fbest, calculations \
            = nmead(func,
                    [[10., 20., -5.0],
                     [7.0, 3.1, -6.0],
                     [7.1, 3.3, -6.1],
                     [8.0, 15.0, -10.0]],
                    tol = 1e-6)

        self.assertVectorEqual( xbest, [0, 0, 0] )
        

    def test_rozenbrock2d(self):
        xbest, fbest, calculations \
            = nmead(rozenbrock,
                    [[.1, .0],
                     [.0, .1],
                     [-.1,.0]],
                    tol = 1e-6)
        self.assertVectorEqual( xbest, [1.0, 1.0] )

    def test_double_rozen(self):
        def rozen4(x):
            return rozenbrock(x[:2]) + rozenbrock(x[2:]) * 2

        xbest, fbest, calculations \
            = nmead(rozen4,
                    [[.0,.0,.0,.0]] + eye(4),
                    tol = 1e-6)

        self.assertVectorEqual( xbest, [1.0, 1.0, 1.0, 1.0] )
        
    

    def test_fmin_nmead(self):
        xbest, fbest, calculations \
            = fmin_nmead(rozenbrock, [0.0,0.0], tol = 1e-6)
        self.assertVectorEqual( xbest, [1.0, 1.0] )

        xbest, fbest, calculations \
            = fmin_nmead(rozenbrock, [10.0,10.0], tol = 1e-6)
        self.assertVectorEqual( xbest, [1.0, 1.0] )







