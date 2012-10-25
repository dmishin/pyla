"""Linear algebra algorithms in pure python"""
try:
    assert(xrange)
    #python2
    from itertools import izip
    flatzip = zip
except:
    #python3
    xrange = range
    from functools import reduce
    izip = zip
    def flatzip(*args):
        return list(zip(*args))

import math
import random
from pyla.numeric_context import FloatContext

def to_context_mat( iterable, context=FloatContext ):
    """Convert matrix to the given context"""
    to_float = context.from_int
    return [[to_float(x) for x in row] for row in iterable]

def to_context_vec( iterable, context=FloatContext ):
    """Convert vector to the given context"""
    to_float = context.from_int
    return [to_float(x) for x in iterable]


def lcombine( v1, v2, k1, k2 ):
    """Linear combination of 2 vectors (lists)"""
    return [ x*k1 + y*k2 for (x,y) in izip(v1,v2) ]
def combine_mat( m1, m2, k1, k2 ):
    """Linear combination of 2 matrices"""
    return [ lcombine( r1, r2, k1, k2 )for r1, r2 in izip(m1, m2) ]
def vec_eq( v1, v2, tol=1e-14, context = FloatContext ):
    assert( len(v1) == len(v2) )
    fabs = context.fabs
    for x, y in izip(v1,v2):
        if fabs(x-y)>tol: return False
    return True

def vec_sum( v1, v2 ):
    return [ x+y for x,y in izip(v1,v2) ]

def vec_diff( v1, v2 ):
    return [ x-y for x,y in izip(v1,v2) ]

def mat_eq( m1, m2, tol=1e-14 ):
    for x,y in izip(m1,m2):
        if not vec_eq(x,y,tol): return False
    return True

def vec_scale( v, k ):
    return [ x*k for x in v ]

def vec_scale_inplace( v, k ):
    for i in xrange(len(v)):
        v[i] *= k
def vec_add_inplace( v1, v2 ):
    for i in xrange( len(v1) ):
        v1[i] += v2[i]
def mat_scale( m, k ):
    return [ vec_scale(row, k) for row in m ]

def mat_scale_inplace( m, k ):
    for row in m:
        vec_scale_inplace( row, k )

def mat_sum( m1, m2 ):
    return [ vec_sum(r1, r2) for r1, r2 in izip(m1,m2)]
def mat_add_inplace( m1, m2 ):
    for row1, row2 in izip(m1,m2):
        vec_add_inplace( row1, row2 )
def mat_diff(m1, m2):
    return [ vec_diff(r1, r2) for r1, r2 in izip(m1,m2)]

def eye(n, context = FloatContext):
    """Identity matrix"""
    one,zero = context.one, context.zero
    return [_ort(i,n,zero,one) for i in xrange(n)]

def _ort(i,n,zero,one):
    """i'th n-dimensional ort"""
    o = [zero] * n
    o[i] = one
    return o

def mul_row_mat( row, m ):
    """Product of a row matrix (single list) and rectangular matrix"""
    assert( len(row) == len(m) )
    return reduce( vec_sum, 
                  (vec_scale(m_i, r_i) for r_i, m_i in izip(row, m)))

def mmul( m1, m2 ):
    """Matrix multiplication"""
    return [ mul_row_mat( m1_row, m2) for m1_row in m1 ]

def mv_mul( m, v ):
    """M*v"""
    assert( len(m[0]) == len(v) )
    return [dot( m_i, v ) for m_i in m ]

def dot( v1, v2 ):
    return sum( x*y for x,y in izip(v1,v2) )

def transpose(m):
    return [list(row) for row in izip( *m )]

def shape_mat(m):
    return len(m), len(m[0])

def diag(M,idx=0):
    """Returns i'th diagonal.
    i=0 - main diagonal;
    i>0 - above main diagonal; i<0 - below it."""
    n, m = shape_mat(M)
    if idx >= 0:
        return [ M[i][i+idx] for i in xrange( min( n, m-idx ) ) ]
    else:
        return [ M[i-idx][i] for i in xrange( min( n+idx, m ) ) ]

def set_diag(M,d,idx=0):
    """Returns i'th diagonal.
    i=0 - main diagonal;
    i>0 - above main diagonal; i<0 - below it."""
    n, m = shape_mat(M)
    if idx >= 0:
        for i, di in enumerate( d ):
            M[i][i+idx] = di
    else:
        for i, di in enumerate( d ):
            M[i-idx][i] = di

def from_diag(d, context = FloatContext):
    """Make matrix from its diagonal"""
    n = len(d)
    S = zeros(n,n,context)
    set_diag(S,d)
    return S

def zeros(n,m,context = FloatContext):
    zero = context.zero
    return [[zero for _ in xrange(m)] for __ in xrange(n)]

def copy_mat( m ):
    return [ row[:] for row in m ]

def flip_ud(m):
    return m[::-1]

def flip_lr(m):
    return [ row[::-1] for row in m]

def vec_norm2( v, context = FloatContext ):
    return context.sqrt(sum( x**2 for x in v ))

def normalized_vec(v, context = FloatContext):
    n = vec_norm2( v, context)
    return [ x / n for x in v ]

def rand_mat(n,m,context=FloatContext):
    r = random.random
    to_context = context.from_int
    return [[to_context(r()) for _ in xrange(m)] for __ in xrange(n)]

def show_mat(M):
    return "\n".join( str(row) for row in M )


################################################################################
# Triangular matrices
################################################################################
def extract_ltri( m, context = FloatContext ):
    """Get left triangular matrix"""
    zero = context.zero
    n,n_ = shape_mat(m)
    return [[ m[i][j] if i >= j else zero 
              for j in xrange(n_)] 
            for i in xrange(n)]


def extract_ltri( m, context = FloatContext ):
    """Get left triangular matrix"""
    rows, cols = shape_mat(m)
    return [ row[:i+1] + [context.zero]*(cols - i - 1) 
             for i, row in enumerate(m) ]

def extract_utri( m, context = FloatContext ):
    """Get upper triangular matrix"""
    rows, cols = shape_mat(m)
    return [ [context.zero]*i + row[i:]
             for i, row in enumerate(m) ]
################################################################################
# Matrix inverse
################################################################################

def _unscale_vec_inplace( v, k, i0 ):
    for i in xrange(i0, len(v)):
        v[i] /= k

def _add_vec_scaled( v, dv, k, i0 ):
    for i in xrange(i0, len(v)):
        v[i] += dv[i] * k

def inverse( m, context = FloatContext ):
    """Matrix inverse, using Gauss-Jordan elimination"""
    n,n_ = shape_mat(m)
    m = copy_mat(m) #the transformation is destructing, so make a copy.

    assert (n==n_) #matris xhould be square
    fabs = context.fabs
    one, zero = context.one, context.zero
    def find_pivot( m, i ):
        j_pivot = i
        m_pivot = fabs(m[i][i])
        for j in xrange( i+1, n ):
            m_ji = fabs(m[j][i])
            if m_ji > m_pivot:
                m_pivot, j_pivot = m_ji, j
        return j_pivot
                
    
    im = eye( n, context = context ) #original inverse

    #Now apply same linear transformations to m and im; to make m equal to eye
    for i in xrange(n):
        #Choose a pivot: a row with maximal element [i, j]
        pivot_row = find_pivot( m, i )
        #Put pivot row to the current position
        if pivot_row != i:
            m[i], m[pivot_row] = m[pivot_row], m[i]
            im[i], im[pivot_row] = im[pivot_row], im[i]
        #normalize pivot row
        m_ii = m[i][i]
        _unscale_vec_inplace( m[i], m_ii,   i+1 )
        #m[i][i] = one                             #do not try to un-scale self; this should always give 1.
        _unscale_vec_inplace( im[i],    m_ii, 0 )
        #make other rows to have zero in current column
        for j in xrange(n):
            if j == i : continue #skipping self
            k = m[j][i]
            _add_vec_scaled( m[j], m[i], -k, i+1 )
            #m[j][i] = zero                        #do not try to eliminate i'th element; 0 alwasy should go here.
            _add_vec_scaled( im[j], im[i], -k, 0 )
    #Done!
    #m should contain eye matrix at this step
    return im

