import numpy as np


def cg(xk, A, b):
    """
    Conjugate gradient taken from the function _minimize_newtoncg in the package scipy.optimize.optimize.

     See https://github.com/scipy/scipy/blob/v1.1.0/scipy/optimize/optimize.py line 1471
     """

    float64eps = np.finfo(np.float64).eps

    cg_maxiter = 20*len(xk)

    maggrad = np.add.reduce(np.abs(b))
    eta = np.min([0.5, np.sqrt(maggrad)])
    termcond = eta * maggrad

    xsupi = np.zeros(len(xk))

    ri = -b
    psupi = -ri
    i = 0
    dri0 = np.dot(ri, ri)

    for k2 in range(cg_maxiter):
        if np.add.reduce(np.abs(ri)) <= termcond:
            break
        Ap = np.dot(A, psupi)
        # check curvature
        Ap = np.asarray(Ap).squeeze()  # get rid of matrices...
        curv = np.dot(psupi, Ap)
        if 0 <= curv <= 3 * float64eps:
            break
        elif curv < 0:
            if (i > 0):
                break
            else:
                # fall back to steepest descent direction
                xsupi = dri0 / (-curv) * b
                break
        alphai = dri0 / curv
        xsupi = xsupi + alphai * psupi
        ri = ri + alphai * Ap
        dri1 = np.dot(ri, ri)
        betai = dri1 / dri0
        psupi = -ri + betai * psupi
        i = i + 1
        dri0 = dri1  # update np.dot(ri,ri) for next time.

    return xsupi
