import unittest
from pyla.core import *
from pyla import svd


class TestHouseholderQR(unittest.TestCase):
    def test_basic_qr(self):
        N = 5
        for i in range(100):
            m = rand_mat(N,N)
            q,r = svd.qr_householder( m )
            self.assertTrue( mat_eq( mmul( q, transpose(q)),
                                     eye(N), tol=1e-5 ) ) #is orthogonal
            self.assertTrue( mat_eq( r, extract_ltri(r), tol=1e-5 ) ) #is triangular
            self.assertTrue( mat_eq( mmul( q, r),
                                     m, tol=1e-5 ) ) #is right
        
class TestHouseholderTools(unittest.TestCase):
    def test_householder_tfm(self):
        
        a = [[1.0, 2.0, 3.0],
             [2.0, 0.0, -1.0]]
        
        svd._householder_tfm_rows_tail_inplace( a, [1.0, 0.0, 0.0] )
        self.assertTrue( mat_eq( a, 
                                 [[-1.0, 2.0, 3.0],
                                  [-2.0, 0.0, -1.0]] ) )
        svd._householder_tfm_rows_tail_inplace( a, [1.0, 0.0, 0.0] )
        self.assertTrue( mat_eq( a, 
                                 [[1.0, 2.0, 3.0],
                                  [2.0, 0.0, -1.0]] ) )
    
    def test_householder_tfm_tail_numpy(self):
        """Big test on arbitrary householder transformations"""
        import numpy
        N, M = 2,3
        for i in xrange(100):
            a = numpy.random.rand( N,M) - 0.5
            v = numpy.random.rand( M ) - 0.5
            vn = numpy.array(normalized_vec( v ))
            
            H = numpy.eye(M) - 2*numpy.dot( vn.reshape((M,1)), vn.reshape((1,M)) ) # H = E - 2*vv'
            a1 = numpy.dot(a, H)

            svd._householder_tfm_rows_tail_inplace( a, vn )
            self.assertTrue( mat_eq( a1, a, tol=1e-5 ) )

    def test_householder_tfm_head_numpy(self):
        """Big test on arbitrary householder head transformations"""
        import numpy
        N, M = 2,3
        for i in xrange(100):
            a = numpy.random.rand( N,M) - 0.5
            v = numpy.random.rand( M ) - 0.5
            vn = numpy.array(normalized_vec( v ))
            
            H = numpy.eye(M) - 2*numpy.dot( vn.reshape((M,1)), vn.reshape((1,M)) ) # H = E - 2*vv'
            a1 = numpy.dot(a, H)

            svd._householder_tfm_rows_head_inplace( a, vn )
            self.assertTrue( mat_eq( a1, a, tol=1e-5 ) )

    def test_householder_bring1(self):
        v = [1.0, 2.0, 3.0]
        b = svd._householder_bring_vector(v, 0)
        m = copy_mat( [v] )
        svd._householder_tfm_rows_tail_inplace( m, b )

        self.assertTrue( abs(m[0][0]) > 0 )
        self.assertTrue( abs(m[0][1]) < 1e-5 )
        self.assertTrue( abs(m[0][2]) < 1e-5 )

        self.assertTrue( abs(abs(m[0][0]) - vec_norm2(v)) < 1e-5 )

    def test_householder_bring2(self):
        v = [1.0, 2.0, 3.0]
        b = svd._householder_bring_vector(v, -1)
        m = copy_mat( [v] )
        svd._householder_tfm_rows_tail_inplace( m, b )

        self.assertTrue( abs(m[0][0]) < 1e-5 )
        self.assertTrue( abs(m[0][1]) < 1e-5 )
        self.assertTrue( abs(m[0][2]) > 0  )

        self.assertTrue( abs(abs(m[0][2]) - vec_norm2(v)) < 1e-5 )

class TestSVD(unittest.TestCase):
    def test_bidiagonal_tfm(self):
        #Test that bidiagonal transform works
        #Take a sample matrix:
        M = [[1.0, 2.0, 3.0, 4.0],
             [3.0, 5.0, -1.0, 0.0],
             [2.0, -2.0, 1.0, 1.0],
             [-5.0, -2.0, -1.0, 0.0]]
        #And calculate it's bidiagonal transform:
        U,B,V = svd.bidiagonal_transform( M )
        #ensure that U and V matrices are orthogonal
        self.assertTrue( mat_eq( mmul( U, transpose(U) ),
                                 eye( 4 ),
                                 tol = 1e-5) )
        self.assertTrue( mat_eq( mmul( V, transpose(V) ),
                                 eye( 4 ),
                                 tol = 1e-5) )
        #ENsure that decomposition is correct.
        M1 = mmul(mmul( U, B ), V)
        self.assertTrue( mat_eq( M, M1, tol = 1e-5) )

        #Ensure that B matrix is bidiagonal
        self.assertTrue( mat_eq( B,
                                 extract_ltri( B ),
                                 tol = 1e-5 ) )
        self.assertTrue( vec_eq( diag(B, -2), 
                                 [0.0,0.0] ) )
        self.assertTrue( vec_eq( diag(B, -3), 
                                 [0.0] ) )
        
    def assertOrthogonal( self, m, tol=1e-5 ):
        e = eye (len(m))
        self.assertTrue( mat_eq( mmul( m, transpose(m)),
                                 e,
                                 tol ) )
    def test_svd(self):
        #Test that SVD works
        #Take a sample matrix:
        M = [[1.0, 2.0, 3.0, 4.0],
             [3.0, 5.0, -1.0, 0.0],
             [2.0, -2.0, 1.0, 1.0],
             [-5.0, -2.0, -1.0, 0.0]]
        #And calculate it's bidiagonal transform:
        U,s,V = svd.svd( M )
        #ENsure that decomposition is correct.
        M1 = reduce( mmul, [U, from_diag(s), V] )
        self.assertTrue( mat_eq( M, M1, 1e-5 ) )
        #Ensure that matrices are orthogonal
        self.assertOrthogonal( U )
        self.assertOrthogonal( V )


    def test_svd_singular(self):
        #Let's make a singular matrix M:
        x = [[1.0, 2.0, 3.0, 4.0],
             [-5.0, -2.0, -1.0, 0.0]]
        M = mmul(transpose(x), x)
        #And see, what it's SVD looks like:
        U,s,V = svd.svd(M)
        #ensure that rank is correct
        self.assertTrue( sum( 1 if abs(si) > 1e-5 else 0
                              for si in s ) == 2 ) #rank must be 2
        #ENsure that decomposition is correct.
        M1 = reduce( mmul, [U, from_diag(s), V] )
        self.assertTrue( mat_eq( M, M1, 1e-5 ) )
        #Ensure that matrices are orthogonal
        self.assertOrthogonal( U )
        self.assertOrthogonal( V )
        
    def test_rank_square(self):
        #Ranks of square matrices
        x = [[1.0, 2.0, 3.0, 4.0],
             [-5.0, -2.0, -1.0, 0.0]]
        self.assertEqual( svd.rank( mmul( transpose(x), x) ),
                          2 )
        x = [[1.0, 2.0, 3.0, 4.0] ]
        self.assertEqual( svd.rank( mmul( transpose(x), x) ),
                          1 )
        x = [[1.0, 2.0, 3.0, 4.0],
             [1.0, 0.0,0.0,0.0],
             [-5.0, -2.0, -1.0, 0.0]]
        self.assertEqual( svd.rank( mmul( transpose(x), x) ),
                          3 )

class TestPseudoInverse(unittest.TestCase):
    def test_pinv_full_rank(self):
        M = [[1.0, 2.0, 3.0, 4.0],
             [3.0, 5.0, -1.0, 0.0],
             [2.0, -2.0, 1.0, 1.0],
             [-5.0, -2.0, -1.0, 0.0]]
        self.assertEqual( svd.rank(M), 4 ) #Rank is full
        #print (svd.svd(M)[1])
        IM = svd.pinv(M)
        #print (svd.svd(IM)[1])
        IIM = svd.pinv(IM)
        #print ("----")
        #print (show_mat( mat_diff( M, IIM )))
        #print (show_mat( mmul(M, IM) ) )
        self.assertTrue( mat_eq( mmul(M, IM),
                                 eye(4),
                                 1e-5 ) ) #Inverse is right
        self.assertTrue( mat_eq( M, IIM, 1e-5 ) ) #double pinv is the same as original

        
if __name__ == '__main__':
    unittest.main()


