from .Direction import Direction
from .helpers import cg


class Hessian(Direction):

    def __init__(self, **kwargs):
        Direction.__init__(self, **kwargs)

    def compute_func_and_derivatives(self, mult, batch, full_size):

        if batch != full_size or self.batch_changed:
            # Set the sample for the batch
            sample = self.biogeme.database.data.sample(n=batch, replace=False)
            self.biogeme.theC.setData(sample)

        def grad_hess(x, B):
            tmp = self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=True)

            ret = []
            for i in [1,2]:
                ret.append(mult * tmp[i])

            return ret

        fprime = lambda x: mult * self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=False)[1]
        f = lambda x: mult * self.f(x)

        return f, fprime, grad_hess

    def compute_direction(self, xk, gk, Bk):

        return cg(xk, Bk, -gk)

    def to_str(self):
        return "Hessian"
