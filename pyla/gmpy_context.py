from pyla.numeric_context import NumericContext
import gmpy

def GMPYContext( prec ):
    """Create new GMPY context.
    GMPY settings can be changed, so re-creating this context as needed
    """
    eps = gmpy.mpf(2, prec) ** (-prec + 3)
    return NumericContext(
        one=gmpy.mpf(1,prec),
        zero=gmpy.mpf(0, prec),
        fabs=abs,
        sqrt=gmpy.fsqrt,
        from_int = lambda x: gmpy.mpf(x, prec),
        eps = eps
        )


if __name__=="__main__":
    import pyla.core as pylinalg
    from pyla import svd

    context = GMPYContext(300)
    
    m = pylinalg.to_context_mat(pylinalg.rand_mat(4,4),
                                context=context)

    u,s,v = svd.svd( m, context=context, tol = context.from_int(1e-150) )
    print (pylinalg.show_mat(u))
    print (pylinalg.show_mat(v))
    print (s)

    m1 = reduce( pylinalg.mmul, [u, pylinalg.from_diag(s,context=context),
                                 v] )
    print ("-------------------------------------")
    print (pylinalg.show_mat( pylinalg.mat_diff( m, m1 ) ))
