import sys

class Type:

    def __init__(self, **kwargs):

        # Parameters from the kwargs
        self.full_size = kwargs.get('full_size', None)
        self.biogeme = kwargs.get('biogeme', None)
        self.verbose = kwargs.get('verbose', False)

    def update_batch(self):
        pass

    def compute_alpha(self):
        pass

    def to_str(self):
        return "type"

    def _write(self, msg):
        sys.stderr.write(msg)

