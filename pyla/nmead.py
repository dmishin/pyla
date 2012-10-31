from pyla.core import *
import random
from itertools import izip

def sumv(vecs):
    s = vecs[0][:]
    for v in vecs[1:]:
        vec_add_inplace(s,v)
    return s

def mean(vecs):
    return vec_scale(sumv(vecs),
                     1.0/len(vecs))
    
def maxidx(vec):
    imin = None
    val = None
    for idx, x in enumerate(vec):
        if val == None or x>val:
            imin = idx
            val = x
    return val, imin

def minidx(vec):
    imin = None
    val = None
    for idx, x in enumerate(vec):
        if val == None or x<val:
            imin = idx
            val = x
    return val, imin

def vec_range(vecs):
    return [ max(xis) - min(xis) for xis in izip(*vecs) ]
    
def nmead(poly, func, alpha=1.0, beta=2.0, gamma=0.5, delta=0.5, tol = 1e-6, max_steps = 1000):
    steps = 0
    calculations = 0
    
    assert(len(poly) - 1 == len(poly[0]))
    
    def refl(xc, x):
        return lcombine(xc, x, 1+alpha, -alpha)
        
    def expand(xc, x):
        return lcombine(xc, x, 1-beta, beta)
    def shrink(xc, x):
        return lcombine(xc, x, 1-gamma, gamma)
    
    def is_finish(points):
        return sum(vec_range(points)) < tol
    
    vals = map(func, poly)
    calculations += len(poly)
    
    
    while steps <= max_steps and not is_finish(poly):
        f_worst, worst_i = maxidx(vals)
        x_worst = poly[worst_i]
        print "X worst: %s, F(Xw)=%s"%(str(x_worst), f_worst)
        del poly[worst_i]
        
        f_best = min(vals)

        x_c = mean(poly)
        x_r = refl(x_c, x_worst)
        f_r = func(x_r)
        calculations += 1

        print "  Center:",str(x_c)
        print "  X_r=%s, F_r=%s"%(str(x_r), f_r)
        print "  Fbest=%s"%(f_best)
        
        needScale = False
        if f_r < f_worst: #reflection succeed
            x_e = expand(x_c, x_r)
            f_e = func(x_e)
            calculations += 1
            print "  trying expand... F_e=%s"%f_e,
            if f_e < f_r: #expansion succeed
                x_next = x_e
                f_next = f_e
                #print "Expand:", 
            else: #expaansion fail
                x_next = x_r
                f_next = f_r
                print "Flip, noexpand:", 
        else : #reflection partial success
            print "  trying contract...",
            x_s = shrink(x_c, x_worst)
            f_s = func(x_s)
            calculations += 1
            if f_s < f_worst: #shrink success
                x_next = x_s
                f_next = f_s
                print "ShrinK:", 
            else:
                x_next = x_worst
                f_next = f_worst
                needScale = True
        
        poly.append(x_next)
        vals.append(f_next)

        if needScale:
            for i in range(len(poly)):
                poly[i] = lcombine(x_c, poly[i], 1-delta, delta)
            vals = map(func, poly)
            calculations += len(poly)
            print "Total scale:", 
        print (poly)
        print "------------------------"
        steps += 1
    #something found
    xbest = mean(poly)
    fbest = func(xbest)
    calculations += 1
    #print "Total calculations: %d, total steps: %d"%(calculations, steps)
    return xbest, fbest, steps, calculations

