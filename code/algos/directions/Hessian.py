from .Direction import Direction
from .helpers import cg
from ..helpers import back_to_bounds


class Hessian(Direction):

    def __init__(self, **kwargs):
        Direction.__init__(self, **kwargs)

    def compute_func_and_derivatives(self, batch, full_size):

        if batch != full_size or self.batch_changed:
            # Set the sample for the batch
            sample = self.biogeme.database.data.sample(n=batch, replace=False)

            self.biogeme.theC.setData(sample)

        def grad_hess(x, B):
            x = back_to_bounds(x, self.bounds)

            tmp = self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=True)

            ret = []
            for i in [1,2]:
                ret.append(self.mult * tmp[i])

            return ret

        def fprime(x):
            x = back_to_bounds(x, self.bounds)
            return self.mult * self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=False)[1]

        def f(x):
            x = back_to_bounds(x, self.bounds)
            return self.mult * self.f(x)

        return f, fprime, grad_hess

    def compute_func_and_derivatives2(self, batch, full_size):

        if batch != full_size or self.batch_changed:
            # Set the sample for the batch
            sample = self.biogeme.database.data.sample(n=batch, replace=False)
            # Save the sample index for TR-Bastin
            self.sample_idx = sample.index.tolist()

            self.biogeme.theC.setData(sample)

        if batch == full_size and self.batch_changed:
            self.sample_idx = self.biogeme.database.data.index.tolist()

        def grad_hess(x, B):
            x = back_to_bounds(x, self.bounds)

            tmp = self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=True)

            ret = []
            for i in [1,2]:
                ret.append(self.mult * tmp[i])

            return ret

        def fprime(x):
            x = back_to_bounds(x, self.bounds)
            return self.mult * self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=False)[1]

        def f(x):
            x = back_to_bounds(x, self.bounds)
            return self.mult * self.f(x)

        return f, fprime, grad_hess

    def compute_direction(self, xk, gk, Bk):

        return cg(xk, Bk, -gk)

    def to_str(self):
        return "Hessian"
