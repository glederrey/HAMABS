import sys

class Direction:

    def __init__(self, **kwargs):
        # Parameters from the kwargs
        self.biogeme = kwargs.get('biogeme', None)
        self.verbose = kwargs.get('verbose', False)
        self.x0 = kwargs.get('x0', None)
        self.bounds = kwargs.get('bounds', None)

        self.f = lambda x: self.biogeme.calculateLikelihood(x)

        self.batch_changed = False

    def compute_full_f(self, x):
        # Set the full data
        self.biogeme.theC.setData(self.biogeme.database.data)

        # Return the value of the function
        return self.f(x)

    def init_hessian(self, x0):
        # Used by BFGS
        return None

    def upd_hessian(self, xk, xk_new, f, fprime, Bk):
        # Used by BFGS and Hybrid
        return Bk

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
