def bisection(f, a, b, tol, **kwds):
    c = (a+b)/2.0
    while (b-a)/2.0 > tol:
        if f(c, **kwds) == 0:
            return c
        elif f(a, **kwds)*f(c, **kwds) < 0:
            b = c
        else:
            a = c
        c = (a+b)/2.0
    return c
