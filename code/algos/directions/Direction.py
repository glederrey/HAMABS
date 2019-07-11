import sys


class Direction:

    def __init__(self, **kwargs):
        # Parameters from the kwargs
        self.biogeme = kwargs.get('biogeme', None)
        self.verbose = kwargs.get('verbose', False)

        self.f = lambda x: self.biogeme.calculateLikelihood(x)

    def compute_full_f(self, x):
        # Set the full data
        self.biogeme.theC.setData(self.biogeme.database.data)

        # Return the value of the function
        return self.f(x)

    def compute_func_and_derivatives(self):
        pass

    def compute_direction(self):
        pass

    def to_str(self):
        return "direction"

    def _write(self, msg):
        sys.stderr.write(msg)
