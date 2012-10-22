from pyla.core import NumericContext
import gmpy

def GMPYContext():
    """Create new GMPY context.
    GMPY settings can be changed, so re-creating this context as needed
    """
    return NumericContext(
        one=gmpy.mpf(1),
        zero=gmpy.mpf(0),
        fabs=abs,
        sqrt=gmpy.fsqrt,
        from_int = gmpy.mpf
        )


if __name__=="__main__":
    import pylinalg
    import svd
    gmpy.set_minprec(300)
    context = GMPYContext()
    
    m = [[gmpy.mpf(x) for x in row] 
         for row in pylinalg.rand_mat(4,4)]

    u,s,v = svd.svd( m, context=context, tol = gmpy.mpf(1e-150) )
    print (pylinalg.show_mat(u))
    print (pylinalg.show_mat(v))
    print (s)

    m1 = reduce( pylinalg.mmul, [u, pylinalg.from_diag(s,context=context),
                                 v] )
    print ("-------------------------------------")
    print (pylinalg.show_mat( pylinalg.mat_diff( m, m1 ) ))
