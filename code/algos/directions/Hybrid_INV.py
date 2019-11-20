from .Direction import Direction
from .helpers import cg
from ..helpers import back_to_bounds
import time

import numpy as np


class Hybrid_INV(Direction):

    def __init__(self, **kwargs):
        Direction.__init__(self, **kwargs)

        self.I = np.eye(len(self.x0))

        self.use_hessian = True

        self.switch = False

        self.perc = kwargs.get('perc_hybrid', 30)

        self.time_hessian = []
        self.batches = []

    def compute_func_and_derivatives(self, batch, normalization, full_size):

        self.batches.append(batch)

        if batch != full_size or self.batch_changed:
            # Set the sample for the batch
            sample = self.biogeme.database.data.sample(n=batch, replace=False)

            self.biogeme.theC.setData(sample)

        def grad_hess(x, B):
            x = back_to_bounds(x, self.bounds)

            if self.use_hessian:
                # Same as Hessian
                start = time.time()
                tmp = self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=True)
                self.time_hessian.append(time.time()-start)

                ret = []
                for i in [1,2]:
                    ret.append(self.mult / normalization * tmp[i])

                return ret
            else:
                # Same as BFGS
                start = time.time()
                tmp = self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=False)
                self.time_hessian.append(time.time()-start)

                # Inverse the Hessian the first time we switch from Hessian step to BFGS
                if self.switch:
                    try:
                        B = np.linalg.inv(B)
                    except:
                        B = np.eye(len(B))
                    self.switch = False

                ret = [self.mult / normalization * tmp[1], B]

                return ret

        def fprime(x):
            x = back_to_bounds(x, self.bounds)
            return self.mult / normalization * self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=False)[1]

        def f(x):
            x = back_to_bounds(x, self.bounds)
            return self.mult / normalization * self.f(x)

        return f, fprime, grad_hess

    def init_hessian(self, x0):

        return self.I

    def compute_direction(self, xk, gk, Bk):
        if self.use_hessian:
            return cg(xk, Bk, -gk)
        else:
            return -np.dot(Bk, gk)

    def upd_hessian(self, xk, xk_new, f, fprime, Bk, gk):
        if self.use_hessian:
            return Bk
        else:
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

    def update_dir(self, batch, full_size):

        if self.batch_changed:
            # We need to test if we want to switch the algorithm or not

            # Very simple rule
            if batch >= self.perc*full_size/100 and self.use_hessian:
                if self.verbose:
                    self._write("  Change from Newton step to BFGS!\n")

                # We need to know when we switched to make sure that we will inverse the Hessian
                self.switch = True

                self.use_hessian = False

    def to_str(self):
        return "Hybrid"



