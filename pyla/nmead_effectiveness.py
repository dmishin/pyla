from pyla.nmead import nmead, fmin_nmead
from random import random, seed

def rozenbrock(z):
    return (1-z[0])**2+100.0*(z[1]-z[0]**2)**2

def init_dataset(N=1000, rseed=1):
    seed(rseed)
    
    initials = [random_poly(2) for _ in xrange(N)]

    return initials

def random_poly(n):
    def random_vec():
        return [random()*10 - 5 for _ in xrange(n)]
    return [random_vec() for _ in xrange(n+1)]

def measurer(N=1000, rseed=1, tol=1e-6):
    dataset = init_dataset(N, rseed)
    def measure(abgd):
        calls_total = 0
        for poly in dataset:
            x0, f0, calls = nmead(rozenbrock, poly, tol=tol, abgd=abgd)
            calls_total += calls
        return float(calls_total) / len(dataset)
    return measure



msr = measurer(N=100000)
print "Effectiveness:", msr((1.0, 2.0, 0.5, 0.5))

print "searching for the best convergence"
abgd_best, cbest, calls = fmin_nmead( msr, [1.0, 2.0, 0.5, 0.5], dx=0.1, tol=1e-3 )
print "After %d calls, best parameters are: %s"%(calls, abgd_best)
print "Best value is:", cbest
