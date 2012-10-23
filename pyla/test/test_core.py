import unittest
from pyla.core import *
import math

class TestVectorFunctions(unittest.TestCase):

    def setUp(self):
        pass
    def test_vec_eq(self):
        self.assertTrue( vec_eq( [1.0], [1.0] ) )
        self.assertTrue( vec_eq( [1.0,2.0], [1.0,2.0]) )
        self.assertFalse( vec_eq( [1.0,2.0], [1.0,2.1]) )

    def test_lcombine(self):
        a = [1.0,2.0,3.0]
        b = [2.0,2.0,3.0]
        c = lcombine(a,b,2.0,-1.0)
        self.assertTrue( vec_eq( c, [ 0.0, 2.0, 3.0 ] ) )
    
    def test_vec_sum(self):
        self.assertTrue( vec_sum( [1.0, 2.0], [3.0,5.0] ),
                         [4.0, 7.0] )
    def test_vec_diff(self):
        self.assertTrue( vec_diff( [1.0, 2.0], [3.0,5.0] ),
                         [-2.0, -3.0] )
    def test_vec_scale(self):
        self.assertTrue( vec_scale( [2.0, 1.0], 3.0 ),
                         [6.0, 3.0] )
    def test_vec_norm(self):
        v = [1.0,2.0,3.0]
        self.assertTrue( abs( vec_norm2( v ) - math.sqrt( 1 + 4 + 9 ) ) < 1e-5 )

    def test_vec_normalize(self):
        v = [1.0,2.0,3.0]
        nrm = vec_norm2(v)
        vn = normalized_vec( v )
        self.assertTrue( abs( vec_norm2( vn ) - 1 ) < 1e-5 )
        self.assertTrue( vec_eq( vec_scale(vn, nrm), v ) )

class TestMatrixFunctions(unittest.TestCase):
    def setUp(self):
        pass
    def test_mat_eq(self):
        self.assertTrue( mat_eq( [[1.0]], [[1.0]] ) )
        self.assertTrue( mat_eq( [[1.0,2.0],
                                  [3.0,4.0]],
                                 [[1.0,2.0],
                                  [3.0,4.0]] ) )
                         
                         
        self.assertFalse( mat_eq( [[1.0,2.0],
                                   [3.0,4.0]],
                                  [[1.0,2.0],
                                   [-2.0,4.0]] ) )
                         
    def test_mat_scale(self):
        a = [[1.0,2.0],[3.0,4.0]]
        b = [[2.0,4.0],[6.0,8.0]]
        self.assertTrue( mat_eq(b, mat_scale(a,2.0)) )

    def test_mat_sum_diff(self):
        a = [[1.0, 2.0],[3.0,4.0]]
        b = [[1.0,-1.0],[3.0,2.0]]
        c = [[2.0, 1.0],[6.0,6.0]]
        d = [[0.0, 3.0],[0.0,2.0]]
        self.assertTrue( mat_eq(c, mat_sum(a,b)))
        self.assertTrue( mat_eq(d, mat_diff(a,b)))

    def test_mmul(self):
        import numpy
        for i in xrange(100):
            a = numpy.random.random((5,5))
            b = numpy.random.random((5,5))
            ab_numpy = numpy.dot(a,b)
            ab_pyla = mmul(a,b)
            self.assertTrue( mat_eq( ab_numpy, ab_pyla ) )

        for i in xrange(100):
            a = numpy.random.random((5,7))
            b = numpy.random.random((7,5))
            ab_numpy = numpy.dot(a,b)
            ab_pyla = mmul(a,b)
            self.assertTrue( mat_eq( ab_numpy, ab_pyla ) )

    def test_transpose(self):
        c =  [[2.0, 1.0,5.0],[6.0,6.0,5.0]]
        ct = [[2.0, 6.0],
              [1.0, 6.0],
              [5.0, 5.0]]
        self.assertTrue( mat_eq( ct, transpose(c) ) )
        self.assertTrue( mat_eq( c, transpose(ct) ) )
        self.assertTrue( mat_eq( c, transpose(transpose(c) ) ) )

    def test_copy_mat(self):
        c =  [[2.0, 1.0,5.0],[6.0,6.0,5.0]]
        c1 = copy_mat(c)
        self.assertTrue( mat_eq(c, c1) )
        c[1][1] -= 10
        self.assertFalse( mat_eq(c, c1) )

    def test_mv_mul(self):
        import numpy
        for i in xrange(100):
            a = numpy.random.random((5,7))
            b = numpy.random.random((7,))
            ab_numpy = numpy.dot(a,b)
            ab_pyla = mv_mul(a,b)
            
            self.assertTrue( vec_eq( ab_numpy, ab_pyla ) )

    def test_diag(self):
        a = [[1,2,3,4,5],
             [6,7,8,9,0],
             [1,3,5,7,8],
             [2,2,2,2,2]]
        self.assertTrue( vec_eq( diag(a),
                                 [1,7,5,2] ))
        self.assertTrue( vec_eq( diag(a,1),
                                 [2,8,7,2] ))
        self.assertTrue( vec_eq( diag(a,2),
                                 [3,9,8] ))
        self.assertTrue( vec_eq( diag(a,3),
                                 [4,0] ))

        self.assertTrue( vec_eq( diag(a,-1),
                                 [6,3,2] ))
        self.assertTrue( vec_eq( diag(a,-2),
                                 [1,2] ))
        self.assertTrue( vec_eq( diag(a,-3),
                                 [2] ))
    def test_from_diag(self):
        D = from_diag([1,2,3])
        self.assertTrue( mat_eq( D,
                                 [[1,0,0],
                                  [0,2,0],
                                  [0,0,3]] ) )
    def test_set_diag(self):
        a = zeros(4,4)
        set_diag(a, [1,2,3,4], 0)
        set_diag(a, [-1,-2,-3], 1)
        set_diag(a, [5,6,7],-1 )
        self.assertTrue( mat_eq( a,
                                 [[1,-1, 0,  0],
                                  [5, 2,-2,  0],
                                  [0, 6, 3, -3],
                                  [0, 0, 7,  4]] ) )

    def test_inv(self):
        m = [[1., 2., 3.0],[3., 5., 7.0],[1.0, 1.0, 6.0]]
        im = inverse( m )
        iim = inverse( im )
        e = mmul(im,m)
        self.assertTrue( mat_eq( e, eye(3)))
        self.assertTrue( mat_eq( m, iim, 1e-5) )
    


class TestMatrixConstructors(unittest.TestCase):
    def test_eye(self):
        e2 = [[1.0,0.0],[0.0,1.0]]
        self.assertTrue( mat_eq( e2, eye(2)))
    def test_to_context_mat(self):
        #let's take an integer matrix
        m = [[1,2,3],[4,5,6]]
        #and convert it ot the floating context
        mf = to_context_mat( m, context=FloatContext )
        #Then check that conversion was right
        self.assertEqual( shape_mat(m), shape_mat(mf))
        self.assertTrue( mat_eq( m, mf))
        
    def test_to_context_vec(self):
        #let's take an integer vector
        v = [1,2,3,4,5,6]
        #and convert it ot the floating context
        vf = to_context_vec( v, context=FloatContext )
        #Then check that conversion was right
        self.assertEqual( len(v), len(vf))
        self.assertTrue( vec_eq( v, vf))

class TestTrgMatrices(unittest.TestCase):
    def test_extract_ltri(self):
        m = [[1,2,3],[4,5,6],[7,8,9]]
        ml = extract_ltri( m )
        self.assertTrue( mat_eq( ml,
                                 [[1,0,0],[4,5,0],[7,8,9]]))

    def test_extract_utri(self):
        m = [[1,2,3],[4,5,6],[7,8,9]]
        ml = extract_utri( m )
        self.assertTrue( mat_eq( ml,
                                 [[1,2,3],[0,5,6],[0,0,9]]))


        
    
if __name__ == '__main__':
    unittest.main()
