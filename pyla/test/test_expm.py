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
    
    def test_nontrivial(self):
        v = [[1.0, 2.0, -1.0],
             [-1.0, 5.0, 5.0],
             [-2.0, 7.0, 3.0]]
        iv = inverse(v)
        dd = [1.0, 0.5, -1.0]

        d = from_diag( dd )
        ed = from_diag( map( math.exp, dd ) )

        M = reduce( mmul, [v, d, iv] )
        EM_ref = reduce( mmul, [v, ed, iv] )

        EM_exp = expm(M)

        self.assertMatrixEqual( EM_exp, EM_ref )

    def test_expms(self):
        """Calculating of exponents expm(t*M)"""
        
        v = [[1.0, 2.0, -1.0],
             [-1.0, 5.0, 5.0],
             [-2.0, 7.0, 3.0]]
        exp_v = expms(v)

        tt = [0.0, 1.0, -1.0, 2.0, 3.0]
        for t in tt:
            M1 = expm( mat_scale( v, t ) )
            M2 = exp_v( t )
            self.assertMatrixEqual( M1, M2 )
