from .Type import Type
from .helpers import ls_wolfe12


class LineSearch(Type):

    def __init__(self, **kwargs):
        Type.__init__(self, **kwargs)

        self.batch = int(self.full_size)
        self.stocha = False

    def compute_alpha(self, f, fprime, xk, direction, fs):

        if len(fs) == 1:
            old_old_fval = None
        else:
            old_old_fval = fs[-2]

        gk = fprime(xk)

        alpha = ls_wolfe12(f, fprime, xk, direction, gk, fs[-1], old_old_fval)

        return alpha

    def to_str(self):
        return "Line Search"
