################################################################################
# Test functions
################################################################################    

import unittest
from pyla.core import *
from pyla.givens_qr import *

class TestGivensQR(unittest.TestCase):
    def test_basic_qr(self):
        for i in range(100):
            m = rand_mat(5,5)
            q,r = qrl_givens( m )
            self.assertTrue( mat_eq( mmul( q, r),
                                     m ) ) #is right
            self.assertTrue( mat_eq( mmul( q, transpose(q)),
                                     eye(5) ) ) #is orthogonal
            self.assertTrue( mat_eq( r, extract_ltri(r) ) ) #is triangular


    def test_qr_inverse(self):
        for i in range(100):
            m = rand_mat(5,5)
            mi = qr_inverse( m )
            self.assertTrue( mat_eq( mmul( m, mi ),
                                     eye(5), tol=1e-5 ) )
            self.assertTrue( mat_eq( mmul( mi, m ),
                                     eye(5), tol=1e-5 ) )


    def test_triangle_inverse(self):
        for i in range(100):
            m = extract_ltri(rand_mat(5,5))
            self.assertTrue( mat_eq( m, extract_ltri(m)))
            mi = ltri_inverse(m)
            e = mmul(m, mi)
            self.assertTrue( mat_eq( mi, extract_ltri(mi)))
            self.assertTrue( mat_eq( e, eye(5), tol=1e-5 ) )
            
if __name__ == '__main__':
    unittest.main()
