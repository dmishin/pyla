""" Singular value decomposition.

This module implements SVD, usign 2-step algorithm:
1) Bring matrix to the bidiagonal form, using Householder reflections
2) Find bidiagonal SVD, using iterative algorithm
"""
from pyla.core import *
from pyla.givens_qr import givens_inplace #Givens rotation; usie in the iterative part of the algorithm

def _householder_tfm_rows_tail_inplace(m, v):
    """Multiplies M matrix by Householder transformation
    matrix; that is built on vector v.
    If v is shorter than matrix M, it is virtually padded with leading zeros.

    len(v) <= size(m)

    H = E - 2vv'

    M := M * H   #transforming rows of M
    M_i := M_i - 2 M_i v v' == M_i - 2 v * dot(M_i, v)
    """
    rows, cols  = shape_mat( m )
    n1 = cols - len(v)
    for m_i in m:
        k = 2 * dot( m_i[n1:], v )
        m_i[n1:] = [ x - k * y for x,y in zip( m_i[n1:], v ) ]

def _householder_tfm_rows_head_inplace(m, v):
    """Same as tail, but virtual padding is  at the beginning"""
    rows, cols  = shape_mat( m )
    n1 = len(v)
    for m_i in m:
        k = 2 * dot( m_i[:n1], v )
        m_i[:n1] = [ x - k * y for x,y in zip( m_i[:n1], v ) ]

    
def _householder_tfm_cols_tail_inplace(m, v):
    cols, rows  = shape_mat( m )
    n1 = rows - len(v)
    rows_range = xrange( n1, rows )
    for i in xrange(cols): #i is a column index
        m_i = [ m[j][i] for j in rows_range ] #significant part of the i'th column
        
        k = 2 * dot( m_i, v )

        for j, v_j in izip(rows_range,v):
            m[j][i] -= k * v_j

def _householder_bring_vector( v, ort_index, context = FloatContext ):
    """For the given arbitrary vector v,
    returns a vector u; such that:
    householder reflection by u will bring vector v to the i-th axis: Hv = [0, l, 0, 0, ... ]

    Returns vector, or None if reflection is not needed.
    returned vector has unit length
    """
    dv = normalized_vec( v, context )
    dv[ort_index] -= context.one
    n = vec_norm2( dv, context )
    if n < context.eps: return None

    return [x/n for x in dv]

def bidiagonal_transform( m, context=FloatContext, tol=1e-14 ):
    """Bidiagonal decomposition of the matrix.
    Returns 3 matrices: U,BD,V.
    U and V are orthogonal
    BD is bidiagonal, with sub-diagonal non-zero.
    """
    n,n_ = shape_mat( m )
    m = copy_mat(m)
    assert( n==n_) #for now, only square matrices supported
    U,V = eye(n),eye(n)
    for i in xrange(n):
        v = _householder_bring_vector( m[i][i:], 0, context=context )
        if v is not None:
            _householder_tfm_rows_tail_inplace( m, v )
            _householder_tfm_rows_tail_inplace( V, v )

        m = transpose(m)
        if i+1 < n:
            v = _householder_bring_vector( m[i][i+1:], 0, context=context )
            if v is not None:
                _householder_tfm_rows_tail_inplace( m, v )
                _householder_tfm_rows_tail_inplace( U, v )
        m = transpose(m)
    return U, m, transpose(V)
        

def qr_householder( M, context=FloatContext, tol=1e-14 ):
    """QR decomposition, based on Householder reflections"""
    R = transpose(M)
    n, m = shape_mat( R )
    Q = eye(m)
    for i in xrange( n-1, 0, -1 ):
        v = _householder_bring_vector( R[i][:i+1], -1, context )
        if v is None: continue

        _householder_tfm_rows_head_inplace( R, v )
        _householder_tfm_rows_head_inplace( Q, v )

    return Q, transpose(R)
    
def force_zeros(m, context=FloatContext, tol=1e-6):
    """Make near-zero values real zeros"""
    fabs = context.fabs
    zero = context.zero
    return [[x if fabs(x) > tol else zero
             for x in row]
            for row in m]


def _msweep( d, e, U, V, context = FloatContext, eps=1e-14 ):
    """
    Iterative part SVD of a bidiagonal matrix
    Algorithm: Demmel, Kahan
    http://www.math.pitt.edu/~sussmanm/2071Spring08/lab09/index.html
    """
    one = context.one
    zero = context.zero
    fabs = context.fabs
    sqrt = context.sqrt
    max_value = max( (max( fabs(di) for di in d ),
                      max( fabs(ei) for ei in d ) ) )
    tol = eps * max_value
    def annihilating_rotation( a, b ):
        """Calculates rotation matrix [c s; -s c], 
        that annihilates second element of size-2 vector [a;b]"""
        if fabs(b) < tol:
            return (one, zero, a, False)
        elif fabs(a) > fabs(b):
            t = b/a
            t1 = sqrt( one + t**2 )
            c = one/t1
            return (c, t*c, a*t1, True)
        else:
            t = a/b
            t1 = sqrt( one + t**2 )
            s=one/t1
            return (t*s, s, b*t1, True)
    ########################################
    #d = diag( B ) #main diagonal
    #e = diag( B, -1 ) #second diagonal.
    assert( len(d) == 1 + len(e) )
    
    c_old, s_old = one, zero
    c = one
    n = len( d )
    modified = False
    for i in xrange( n-1 ):
        c,s,r,hasRotation1 = annihilating_rotation( c*d[i], e[i] )
        if hasRotation1:
            givens_inplace( U, i, i+1, s, c )
        if i > 0: 
            e[i-1] = r*s_old
        c_old, s_old, d[i], hasRotation2 = annihilating_rotation( c_old*r, d[i+1]*s)
        if hasRotation2:
            givens_inplace( V, i, i+1, s_old, c_old )

        if (not modified) and (hasRotation1 or hasRotation2):
            modified = True

    h = c*d[-1]
    e[-1] = h*s_old
    d[-1] = h*c_old
    return d, e, modified
    
def svd( M, context = FloatContext, tol=None, max_iter=2000):
    """Singular value decomposition, using 2-step algorithm.
    tol - required relative tolerance for the iterative part. 
          If None, (eps) from the context is used.
    max_iter - maximal number of iterations
    """
    if tol is None: tol = context.eps
    #Step 1: get the bidiagonal reduction
    U, BD, V = bidiagonal_transform( M, context=context, tol=tol )
    d = diag(BD)
    e = diag(BD, -1)
    U = transpose(U)
    for i in range(max_iter):
        d,e,modif = _msweep(d,e,U,V, context=context, eps=tol)
        if not modif:
            break
    if modif: print ("Warning: no convergence after %d steps"%(max_iter))
    return transpose(U), d, V


def rank( M, context = FloatContext, tol=1e-5, max_iter=2000):
    """Rank of the matrix"""
    U,s,V = svd(M, context=context, tol = tol / 10, max_iter=max_iter )
    fabs = context.fabs
    return sum( 1 if fabs(si)>tol else 0
                for si in s )

def pinv( M, context = FloatContext, treshold=1e-5, max_iter=2000):
    """Pseudo-inverse.
    Uses singular decomposition inside. All singular values below tol are ignored"""
    U,s,V = svd(M, context=context, max_iter=max_iter )
    one = context.one
    zero = context.zero
    fabs = context.fabs
    abs_treshold = treshold * max( s )
    s_inv = [ one/si if fabs(si) > abs_treshold else zero
              for si in s ]
    return mmul(mmul( transpose(V), from_diag(s_inv, context=context)), transpose(U))



if __name__=="__main__":
    m = rand_mat(5,5)
    
    print(show_mat(m))
    print ()
    U, BD, V = bidiagonal_transform(m)
    print (show_mat(force_zeros(BD)))
    
    m1 = mmul( U, mmul( BD, V))
    print (show_mat(force_zeros(mat_diff(m1, m))))
    print ()
    d = diag(BD)
    e = diag(BD, -1)
    print ("Diagonals:")
    print (d)
    print (e)

    U,V = eye(5), eye(5)
    for i in range(2000):
        #print ("Sweep %d"%(i))
        d,e,modif = _msweep(d,e,U,V)
        if not modif:
            print ("Reached convergence at %d"%i)
            break
    print ("Reminders:")
    print (e)
    print ("Singular values:")
    print (d)
    import numpy.linalg
    print (numpy.linalg.svd(BD)[1])
    

    S = zeros(len(d),len(d))
    set_diag(S,d)
    
    BD1 = reduce( mmul, [transpose(U), S, V] )
    if mat_eq( BD1, BD, 1e-5 ):
        print ("Equality found!!!!!!!!")
                
    
    print ("---------- Full test ----------")
    U,d,V = svd( m )
    print ("Sigval:")
    print (d)
    print ("Restore")
    
    
    m1 = reduce( mmul, [U, from_diag(d), V] )
    print (show_mat(mat_diff(m1,m)))
