from .Type import Type
from .helpers import ls_wolfe12

import numpy as np


class LineSearch(Type):

    def __init__(self, **kwargs):
        Type.__init__(self, **kwargs)

        self.batch = int(self.full_size)
        self.stocha = False

    def compute_alpha(self, f, fprime, xk, direction, fs):

        if len(fs) == 1:
            old_old_fval = None
        else:
            old_old_fval = fs[-2]

        gk = fprime(xk)

        alpha = ls_wolfe12(f, fprime, xk, direction, gk, fs[-1], old_old_fval)

        return alpha

    def update_xk(self, xk, fk, gk, Bk, f, fprime, dir, fs):

        # Compute the direction
        direction = dir.compute_direction(xk, gk, Bk)

        # Compute the value for alpha
        alpha = self.compute_alpha(f, fprime, xk, direction, fs)

        if self.verbose:
            self._write("  ||gk|| = {:.3E}\n".format(np.linalg.norm(gk)))
            self._write("  ||dir|| = {:.3E}\n".format(np.linalg.norm(direction)))
            self._write("  alpha = {:.3E}\n".format(alpha))

        # Update the parameter value
        xk_new = xk + alpha * direction

        return xk_new

    def to_str(self):
        return "Line Search"
