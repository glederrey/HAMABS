from .Direction import Direction
from .helpers import cg
from ..helpers import back_to_bounds

import numpy as np


class Hybrid(Direction):

    def __init__(self, **kwargs):
        Direction.__init__(self, **kwargs)

        self.I = np.eye(len(self.x0))

        self.use_hessian = True

        self.perc = 0.3

    def compute_func_and_derivatives(self, batch, full_size):

        if batch != full_size or self.batch_changed:
            # Set the sample for the batch
            sample = self.biogeme.database.data.sample(n=batch, replace=False)

            self.biogeme.theC.setData(sample)

        def grad_hess(x, B):
            x = back_to_bounds(x, self.bounds)

            if self.use_hessian:
                # Same as Hessian
                tmp = self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=True)

                ret = []
                for i in [1,2]:
                    ret.append(self.mult * tmp[i])

                return ret
            else:
                # Same as BFGS
                tmp = self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=False)

                ret = [self.mult * tmp[1], B]

                return ret

        def fprime(x):
            x = back_to_bounds(x, self.bounds)
            return self.mult * self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=False)[1]

        def f(x):
            x = back_to_bounds(x, self.bounds)
            return self.mult * self.f(x)

        return f, fprime, grad_hess

    def init_hessian(self, x0):

        return self.I

    def compute_direction(self, xk, gk, Bk):
        return cg(xk, Bk, -gk)

    def upd_hessian(self, xk, xk_new, f, fprime, Bk):
        if self.use_hessian:
            return Bk
        else:
            # Have a look at the Wikipedia page for BFGS for this algorithm
            # https://en.wikipedia.org/wiki/Broyden-Fletcher-Goldfarb-Shanno_algorithm
            sk = xk_new - xk

            gk = fprime(xk)
            gk_new = fprime(xk_new)

            yk = gk_new - gk

            try:  # this was handled in numeric, let it remains for more safety
                alpha = 1. / np.dot(yk.T, sk)
            except ZeroDivisionError:
                alpha = 1000.0

            if np.isinf(alpha):  # this is patch for numpy
                alpha = 1000.0

            try:  # this was handled in numeric, let it remains for more safety
                beta = -1 / np.dot(sk.T, np.dot(Bk, sk))
            except ZeroDivisionError:
                beta = 1000.0

            if np.isinf(beta):  # this is patch for numpy
                beta = 1000.0

            u = yk.reshape(len(yk), 1)
            v = np.dot(Bk, sk)
            v = v.reshape(len(v), 1)

            Bk = Bk + alpha * np.dot(u, u.T) + beta * np.dot(v, v.T)

            return Bk

    def update_dir(self, batch, full_size):
        # Very simple rule
        if batch >= self.perc*full_size and self.use_hessian:
            if self.verbose:
                self._write("  Change from Newton step to BFGS!\n")

            self.use_hessian = False

    def to_str(self):
        return "Hybrid"



