import sys
import numpy as np


class Direction:

    def __init__(self, **kwargs):
        # Parameters from the kwargs
        self.biogeme = kwargs.get('biogeme', None)
        self.verbose = kwargs.get('verbose', False)
        self.x0 = kwargs.get('x0', None)
        self.bounds = kwargs.get('bounds', None)
        self.scale = kwargs.get('scale', False)

        self.f = lambda x: self.biogeme.calculateLikelihood(x)

        self.sample_idx = self.biogeme.database.data.index.tolist()

        self.batch_changed = False

        self.mult = 1

        self.I = np.eye(len(self.x0))

    def prep_mult_factor(self, maximize):

        self.mult = 1

        # Inverse the value of the function, the gradient, and the hessian
        if maximize:
            self.mult = -1

    def compute_full_f(self, x):
        # Set the full data
        self.biogeme.theC.setData(self.biogeme.database.data)

        # Return the value of the function
        return self.f(x)

    def init_hessian(self, x0):
        # Used by BFGS
        return self.I

    def upd_hessian(self, xk, xk_new, f, fprime, Bk, gk=None):
        # Used by BFGS and Hybrid
        return Bk

    def compute_final_LL_and_derivatives(self, x, hessian):
        f_val, g_val, B_val, tmp = self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=hessian)
        return f_val, g_val, B_val

    def compute_func_and_derivatives(self):
        pass

    def compute_direction(self):
        pass

    def update_dir(self, batch, full_size):
        pass

    def to_str(self):
        return "direction"

    def _write(self, msg):
        sys.stderr.write(msg)
