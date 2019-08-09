import sys
import numpy as np


def _write(msg):
    sys.stderr.write(msg)


def back_to_bounds(xk, bounds):
    if bounds is not None:
        tmp = []
        for i, x in enumerate(xk):
            tmp.append(min(max(x, bounds[i][0]), bounds[i][1]))

        return np.array(tmp)
    else:
        return xk


def stop_crit(xs, f, grad):

    vals = [np.abs(x*df) for x, df in zip(xs, grad)]

    return np.max(vals)/np.abs(f)
