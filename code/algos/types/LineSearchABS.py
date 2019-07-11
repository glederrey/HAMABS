from .LineSearch import LineSearch
from .ABS import ABS


class LineSearchABS(LineSearch):

    def __init__(self, **kwargs):
        LineSearch.__init__(self, **kwargs)

        self.batch = kwargs.get('batch', int(0.1*self.full_size))
        self.stocha = True

        # Parameters for Batch Size Update
        window = kwargs.get('window', 10)
        thresh_upd = kwargs.get('thresh_upd', 1)
        count_upd = kwargs.get('count_upd', 2)
        factor_upd = kwargs.get('factor_upd', 2)
        self.abs = ABS(window, thresh_upd, count_upd, factor_upd, self.verbose, self.full_size)

    def update_batch(self, it, fk_full):
        self.batch = self.abs.upd(it, fk_full, self.batch)

    def to_str(self):
        return "Line Search ABS"
