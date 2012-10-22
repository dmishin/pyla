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

class NumericContext:
    def __init__(self, one, zero, fabs, sqrt, from_int):
        self.one=one
        self.zero=zero
        self.fabs=fabs
        self.sqrt=sqrt
        self.from_int = from_int
        assert( one * one == one )
        assert( zero * zero == zero )
        assert( one * zero == zero )
        assert( fabs(one) == one )

FloatContext = NumericContext(
    one = 1.0,
    zero = 0.0,
    fabs = abs,
    sqrt = math.sqrt,
    from_int = float 
    )


def lcombine( v1, v2, k1, k2 ):
    """Linear combination of 2 vectors (lists)"""
    return [ x*k1 + y*k2 for (x,y) in izip(v1,v2) ]

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

def mat_scale( m, k ):
    return [ vec_scale(row, k) for row in m ]

def mat_sum( m1, m2 ):
    return [ vec_sum(r1, r2) for r1, r2 in izip(m1,m2)]

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

def rand_mat(n,m):
    r = random.random
    return [[r() for _ in xrange(m)] for __ in xrange(n)]

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
