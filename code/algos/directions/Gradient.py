from .Direction import Direction

from .helpers import back_to_bounds


class Gradient(Direction):

    def __init__(self, **kwargs):
        Direction.__init__(self, **kwargs)

    def compute_func_and_derivatives(self, mult, batch, full_size):

        if batch != full_size or self.batch_changed:
            # Set the sample for the batch
            sample = self.biogeme.database.data.sample(n=batch, replace=False)
            self.biogeme.theC.setData(sample)

        def grad_hess(x, B):
            x = back_to_bounds(x, self.bounds)

            tmp = self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=False)

            ret = [mult*tmp[1], None]

            return ret

        def fprime(x):
            x = back_to_bounds(x, self.bounds)
            return mult * self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=False)[1]

        def f(x):
            x = back_to_bounds(x, self.bounds)
            return mult * self.f(x)

        return f, fprime, grad_hess

    def compute_direction(self, xk, gk, Bk):
        return -gk

    def to_str(self):
        return "Gradient"
