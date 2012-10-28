import unittest
from pyla.core import *
from pyla.expm import *
import math

from pyla_test_case import PylaTestCase
class TestMatrixExponent(PylaTestCase):
    def test_zero(self):
        m = zeros(4,4)
        em = expm(m)
        self.assertMatrixEqual( em, eye(4) )
    def test_eye(self):
        m = eye(4)
        em = expm(m)
        self.assertMatrixEqual( em, mat_scale( m, math.exp(1.0) ) )
    def test_diagonal(self):
        d = [1.0, 2.0, -1.0]
        m = from_diag( d )
        em = expm(m)
        self.assertMatrixEqual( em,
                                from_diag( map(math.exp, d) ) )
    
