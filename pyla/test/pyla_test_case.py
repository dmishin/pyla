import unittest
from pyla.core import *

class PylaTestCase(unittest.TestCase):
    def assertMatrixEqual( self, m1, m2, eps = 1e-5 ):
        self.assertTrue( mat_eq( m1, m2, eps ) )

    def assertVectorEqual( self, v1, v2, eps = 1e-5 ):
        self.assertTrue( vec_eq( v1, v2, eps ) )
