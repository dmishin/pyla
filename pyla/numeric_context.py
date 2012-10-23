import math

class NumericContext:
    def __init__(self, one, zero, fabs, sqrt, from_int, eps):
        self.one=one
        self.zero=zero
        self.fabs=fabs
        self.sqrt=sqrt
        self.from_int = from_int
        self.eps = eps
        assert( one * one == one )
        assert( zero * zero == zero )
        assert( one * zero == zero )
        assert( fabs(one) == one )
        assert( eps > zero )
        assert( one + eps > one )

FloatContext = NumericContext(
    one = 1.0,
    zero = 0.0,
    fabs = abs,
    sqrt = math.sqrt,
    from_int = float,
    eps = 1e-14
    )

