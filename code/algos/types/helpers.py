import numpy as np
from scipy.optimize.linesearch import line_search_wolfe1, line_search_wolfe2, line_search_armijo


def ls_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval):
    """
    Same as line_search_wolfe1, but fall back to line_search_wolfe2 if
    suitable step length is not found, and raise an exception if a
    suitable step length is not found.
    Raises
    ------
    _LineSearchError
        If no suitable step size is found
    """

    ret = line_search_wolfe1(f, fprime, xk, pk, gfk,
                             old_fval, old_old_fval)
    alpha = ret[0]

    if alpha is None or alpha < 1e-12:
        #print('A')
        # line search failed: try different one.
        ret = line_search_wolfe2(f, fprime, xk, pk, gfk,
                                 old_fval, old_old_fval)
        alpha = ret[0]

    if alpha is None or alpha < 1e-12:
        #print('B')
        ret = line_search_armijo(f, xk, pk, gfk, old_fval)
        alpha = ret[0]

    if alpha is None or alpha < 1e-12:
        #print('C')
        alpha = backtracking_line_search(f, gfk, xk, pk)

    return alpha


def backtracking_line_search(func, g, x, step):
    ll = func(x)

    m = np.dot(np.transpose(step), g)

    c = 0.9

    t = -c * m
    tau = 0.5

    alpha = 10

    while True:
        tmp_x = x + alpha * step
        if np.abs(ll - func(tmp_x)) <= alpha * t or alpha < 1e-8:
            return alpha

        alpha = tau * alpha


def line_search(func, x, dir):
    ll = func(x)

    alpha = 5

    while True:
        tmp_x = x + alpha * dir
        if func(tmp_x) < ll or alpha < 1e-8:
            return alpha

        alpha = alpha/2
