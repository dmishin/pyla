from pyla.core import *
import random
from itertools import izip

def sumv(vecs):
    s = vecs[0][:]
    for v in vecs[1:]:
        vec_add_inplace(s,v)
    return s

def mean(vecs):
    s = sumv(vecs)
    vec_scale_inplace( s, 1.0/len(vecs))
    return s

def first( tpl ):
    return tpl[0]

def second( tpl ):
    return tpl[1]

def maxidx( xs ):
    return max( enumerate(xs), key = second )

def minidx( xs ):
    return min( enumerate(xs), key = second )

def nmead(func, poly, abgd = (1.0, 2.0, 0.35, 0.5), tol = 1e-6, max_calls = 10000):
    """Nelder-Mead multivariate optimization.
    Arguments:
     func - Function to optimize.
     poly - Initial polygon. Must be a list of n+1 vectors of size n.
     abgd - 4-tuple of Nelder-Mead method parameters: (alpha, beta, gamma, delta). Default is (1.0, 2.0, 0.35, 0.5), which is usually fine.
     tol - required tolerance by x.
     max_calls - maximal number of calls to the target function to perform.
    Retur value: 3-tuple (x_min, y_min, calls_done)
    """
    
    alpha, beta, gamma, delta = abgd
    def refl(xc, x):
        return lcombine(xc, x, 1+alpha, -alpha)
    def expand(xc, x):
        return lcombine(xc, x, 1-beta, beta)
    def shrink(xc, x):
        return lcombine(xc, x, 1-gamma, gamma)
    def is_converged(points):
        for xis in izip(*points):
            if max(xis)-min(xis) >= tol: return False
        return True
    ###########################################################
    poly = poly[:]
    calculations = 0
    N = len(poly) - 1
    assert( all( len(pi) == N for pi in poly ) )
    
    vals = [func(pi) for pi in poly]
    calculations += N+1
    n_shrinks = 0  #Counter of performed shrinks
    while calculations <= max_calls:
        #####  Sorting  #####
        i_worst, f_worst = maxidx(vals)
        x_worst = poly[i_worst]
        poly[i_worst] = poly[-1]
        vals[i_worst] = vals[-1]
        del poly[-1]
        del vals[-1]
        i_best, f_best = minidx(vals)

        i_worst1, f_worst1 = maxidx(vals)
        x_worst1 = poly[i_worst1]
        ##### Reflection #####
        x_c = mean(poly)
        x_r = refl(x_c, x_worst)
        f_r = func(x_r)
        calculations += 1

        needScale = False
        x_next, f_next = x_r, f_r
        if f_r < f_best: #reflection success
            x_e = expand(x_c, x_r)
            f_e = func(x_e)
            calculations += 1
            if f_e < f_r: #expansion success
                x_next = x_e
                f_next = f_e
        elif f_r < f_worst1: #Partial success
            pass
        else: #no tremendous success, but maybe at least something?
            if f_r < f_worst: #at least better than worst
                #swap xr and the worst point
                x_r, x_worst = x_worst, x_r
                f_r, f_worst = f_worst, f_r
            #now xr is worser than worst
            assert( f_r >= f_worst )
            x_s = shrink(x_c, x_worst)
            f_s = func(x_s)
            calculations += 1
            if f_s < f_worst: #shrink success
                x_next = x_s
                f_next = f_s
                n_shrinks += 1
            else:
                x_next = x_worst
                f_next = f_worst
                needScale = True
        
        poly.append(x_next)
        vals.append(f_next)

        if needScale:
            x_best = poly[i_best]
            for i in xrange(N+1):
                if i != i_best:
                    pi_scaled = lcombine(x_best, poly[i], 1-delta, delta)
                    poly[i] = pi_scaled
                    vals[i] = func(pi_scaled)
            calculations += N
            n_shrinks += N
            
        #Check for convergence every N shrinks.
        if n_shrinks > N:
            n_shrinks = 0
            if is_converged(poly): break

    return poly[i_best], vals[i_best], calculations


    
def fmin_nmead(func, x0, dx=1.0, tol=1e-6):
    """Function minimiser, based on Nelder-Mead method.
    Returns tuple:
    (x_best, y_best, n_calculations)"""
    n = len(x0)
    poly = [x0[:] for _ in xrange(n+1)]
    for i in xrange(n):
        poly[i][i] += dx
    return nmead(func, poly, tol=tol)
