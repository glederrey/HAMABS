from .Direction import Direction
from .helpers import cg

import numpy as np


class Hybrid(Direction):

    def __init__(self, **kwargs):
        Direction.__init__(self, **kwargs)

        self.I = np.eye(len(self.x0))

        self.use_hessian = True

        self.perc = 0.5

    def compute_func_and_derivatives(self, mult, batch, full_size):

        if batch != full_size:
            # Set the sample for the batch
            sample = self.biogeme.database.data.sample(n=batch, replace=False)
            self.biogeme.theC.setData(sample)

        def grad_hess(x, B):

            if self.use_hessian:
                # Same as Hessian
                tmp = self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=True)

                ret = []
                for i in [1,2]:
                    ret.append(mult * tmp[i])

                return ret
            else:
                # Same as BFGS
                tmp = self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=False)

                ret = [mult * tmp[1], B]

                return ret

        fprime = lambda x: mult * self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=False)[1]
        f = lambda x: mult * self.f(x)

        return f, fprime, grad_hess

    def init_hessian(self, x0):

        return self.I

    def compute_direction(self, xk, gk, Bk):
        if self.use_hessian:
            return cg(xk, Bk, -gk)
        else:
            return -np.dot(Bk, gk)

    def upd_hessian(self, xk, xk_new, f, fprime, Bk):
        if self.use_hessian:
            return Bk
        else:
            sk = xk_new - xk

            gk = fprime(xk)
            gk_new = fprime(xk_new)

            yk = gk_new - gk

            try:  # this was handled in numeric, let it remains for more safety
                rhok = 1.0 / (np.dot(yk, sk))
            except ZeroDivisionError:
                rhok = 1000.0
            if np.isinf(rhok):  # this is patch for numpy
                rhok = 1000.0

            A1 = self.I - np.dot(sk[:, np.newaxis], yk[np.newaxis, :]) * rhok
            A2 = self.I - np.dot(yk[:, np.newaxis], sk[np.newaxis, :]) * rhok

            Bk = np.dot(A1, np.dot(Bk, A2)) + (rhok * np.dot(sk[:, np.newaxis], sk[np.newaxis, :]))

            return Bk

    def update_dir(self, batch, full_size, Bk):
        # Very simple rule
        if batch >= self.perc*full_size and self.use_hessian:
            self._write("  Change from Newton step to BFGS!\n")

            self.use_hessian = False

            # Here, we need to invert Bk since we're using the inverse in the BFGS algorithm
            return np.linalg.inv(Bk)
        else:
            return Bk

    def to_str(self):
        return "Hybrid"



