import sys

class Type:

    def __init__(self, **kwargs):

        # Parameters from the kwargs
        self.full_size = kwargs.get('full_size', None)
        self.biogeme = kwargs.get('biogeme', None)
        self.verbose = kwargs.get('verbose', False)
        x0 = kwargs.get('x0', None)
        self.n = len(x0)

    def init_solve(self):
        pass

    def update_xk(self):
        pass

    def update_batch(self, it, fk_full):
        pass

    def to_str(self):
        return "type"

    def _write(self, msg):
        sys.stderr.write(msg)

