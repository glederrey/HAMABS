from .LineSearch import LineSearch
from .ABS import ABS


class LineSearchABS(LineSearch):

    def __init__(self, **kwargs):
        LineSearch.__init__(self, **kwargs)

        self.batch = kwargs.get('batch', int(0.1*self.full_size))


        # The batch size is in percentage
        if self.batch < 1:
            self.batch = int(self.batch*self.full_size)
        else:
            # Make sure it's an int
            self.batch = int(self.batch)


        self.stocha = True

        self.batches = [self.batch]

        # Parameters for Batch Size Update
        update = kwargs.get('update', 'geometric')
        window = kwargs.get('window', 10)
        thresh_upd = kwargs.get('thresh_upd', 1)
        count_upd = kwargs.get('count_upd', 2)
        perc_upd = kwargs.get('perc_upd', 0.1)
        factor_upd = kwargs.get('factor_upd', 2)
        self.abs = ABS(update, window, thresh_upd, count_upd, perc_upd, factor_upd, self.verbose, self.full_size)

    def update_batch(self, it, fk):
        old_batch = self.batch
        self.batch = self.abs.upd(it, fk, self.batch)

        self.batches.append(self.batch)

        if self.batch != old_batch:
            return True
        else:
            return False

    def to_str(self):
        return "Line Search ABS (init batch: {})".format(self.batch)
