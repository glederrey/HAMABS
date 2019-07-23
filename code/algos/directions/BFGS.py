from .Direction import Direction
import numpy as np


class BFGS(Direction):

    def __init__(self, **kwargs):
        Direction.__init__(self, **kwargs)

        self.I = np.eye(len(self.x0))

    def compute_func_and_derivatives(self, mult, batch, full_size):

        if batch != full_size:
            # Set the sample for the batch
            sample = self.biogeme.database.data.sample(n=batch, replace=False)
            self.biogeme.theC.setData(sample)

        def grad_hess(x, B):
            tmp = self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=False)

            ret = [mult*tmp[1], B]

            return ret

        fprime = lambda x: mult * self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=False)[1]
        f = lambda x: mult * self.f(x)

        return f, fprime, grad_hess

    def compute_direction(self, xk, gk, Bk):

        return -np.dot(Bk, gk)

    def init_hessian(self, x0):

        return self.I

    def upd_hessian(self, xk, xk_new, f, fprime, Bk):
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

        Bk = np.dot(A1, np.dot(Bk, A2)) + (rhok * sk[:, np.newaxis] * sk[np.newaxis, :])

        return Bk

    def to_str(self):
        return "BFGS"
