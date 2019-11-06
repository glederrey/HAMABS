from .Direction import Direction
from ..helpers import back_to_bounds
import numpy as np


class BFGS_INV(Direction):
    """

    Same as BFGS, except that we use the update of the inverse of the BFGS instead of the usual update

    """

    def __init__(self, **kwargs):
        Direction.__init__(self, **kwargs)

    def compute_func_and_derivatives(self, batch, full_size):

        if batch != full_size or self.batch_changed:
            # Set the sample for the batch
            sample = self.biogeme.database.data.sample(n=batch, replace=False)

            self.biogeme.theC.setData(sample)

        def grad_hess(x, B):
            x = back_to_bounds(x, self.bounds)

            tmp = self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=False)

            ret = [self.mult / batch * tmp[1], B]

            return ret

        def fprime(x):
            x = back_to_bounds(x, self.bounds)
            return self.mult / batch * self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=False)[1]

        def f(x):
            x = back_to_bounds(x, self.bounds)
            return self.mult / batch * self.f(x)

        return f, fprime, grad_hess

    def compute_direction(self, xk, gk, Bk):

        return -np.dot(Bk, gk)

    def upd_hessian(self, xk, xk_new, f, fprime, Bk, gk=None):
        # Have a look at the Wikipedia page for BFGS for this algorithm
        # Greatly inspired by:
        #     https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py
        sk = xk_new - xk

        if gk is None:
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

        # Bk here is the inverse of the hessian, not the hessian!

        Bk = np.dot(A1, np.dot(Bk, A2)) + (rhok * np.dot(sk[:, np.newaxis], sk[np.newaxis, :]))

        return Bk

    def to_str(self):
        return "BFGS"
