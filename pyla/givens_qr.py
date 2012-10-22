#Simple QR decomposition algorithm
#implemented in pure Python
#and using Givens rotations

try:
    assert(xrange)
    #py2
    from itertools import izip
    flatzip = zip
except ImportError:
    #py3
    from functools import reduce
    izip = zip
    xrange = range
    def flatzip(*args):
        return list(izip(*args))

from pyla.core import flip_ud, transpose, eye, copy_mat, shape_mat, lcombine, mmul, FloatContext

################################################################################
def qrl_givens(M, eps=1e-14, context=FloatContext, inverse_q=False ):
    """QR decomposition of a rectangular matrix. R is right-triangular"""
    n,m = shape_mat(M)
    R = copy_mat(M) #will be used for inplace computations
    Q = eye( n, context=context )
    for j in xrange(min(n,m)-1,0,-1): #from last column to the first
        for i in xrange(j):
            k_sin = R[i  ][j]
            k_cos = R[i+1][j]
            
            k2 = k_sin**2 + k_cos**2
            if k2 < eps: continue
            k = context.sqrt(k2)
            s = k_sin / k
            c = k_cos / k
            
            givens_inplace( Q, i, i+1, -s, c )
            givens_inplace( R, i, i+1, -s, c )
    if inverse_q: return Q,R
    else: return transpose(Q), R

def qrr_givens(M, eps=1e-14, context=FloatContext ):
    """QR decomposition using Givens rotations, returning left-triangular matrix"""
    q,r = qrl_givens(flip_ud(M),eps=eps,context=context)
    return flip_up(q), r

def givens_inplace( M, i, j, s, c ):
    """inplace givens rotation.
    returns M*G
    where G = Givens rotation matrix, having  i-j minor equal to [c s;-s c]
    """
    mi = M[i]
    mj = M[j]
    mi, mj = lcombine( mi, mj, c, s ), lcombine( mi, mj, -s, c )
    M[i] = mi
    M[j] = mj

def ltri_inverse( M, context=FloatContext):
    """Inverse of the left-triangular matrix"""
    R = eye(len(M),context=context)
    for i in xrange(len(M)):
        m_ii = M[i][i] #diagonal element
        r_i = R[i]

        #normalize
        r_i[:i+1] = ( x / m_ii for x in r_i[:i+1])

        #now subtract it from other rows
        for j in xrange(i+1,len(M)):
            r_j = R[j]
            m_ji = M[j][i]

            r_j[:i+1] = ( x - m_ji * y 
                          for x,y in izip( r_j[:i+1], r_i[:i+1] ) )
    return R

def qr_inverse( M, context=FloatContext, eps=1e-14):
    """Matrix inverse, using QR transformation"""
    n,n_ = shape_mat(M)
    assert(n==n_)
    IQ,R = qrl_givens(M, context=context, eps=eps, inverse_q = True)
    IR = ltri_inverse(R, context=context)
    return mmul( IR, IQ )
            

