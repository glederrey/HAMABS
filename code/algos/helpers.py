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
