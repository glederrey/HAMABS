"""
Implementation of the ABS technique
"""

import numpy as np
import sys


class ABS:

    def __init__(self, window=10, thresh_upd=1, count_upd=2, factor_upd=2, verbose=False, max=np.infty):

        self.window = window
        self.thresh_upd = thresh_upd
        self.count_upd = count_upd
        self.factor_upd = factor_upd
        self.verbose = verbose
        self.max = max

        self.full_data = False
        self.count_under_thresh = 0
        self.cum_avg = []
        self.improvements = []
        self.f_vals = []

    def reset(self):
        self.full_data = False
        self.count_under_thresh = 0
        self.cum_avg = []
        self.dcum_avg = []

    def upd(self, curr_iter, f, batch ):

        self.f_vals.append(f)

        self.cum_avg.append(wma(self.f_vals, curr_iter+1, self.window))
        #self.cum_avg.append(ema(self.f_vals, curr_iter + 1, self.window, 5))

        # In case, we don't enter the ifs
        new_batch = batch

        if curr_iter > 0:

            impr = 100*np.abs(self.cum_avg[curr_iter - 1] - self.cum_avg[curr_iter])/np.abs(self.cum_avg[curr_iter - 1])

            self.improvements.append(impr)

            if not self.full_data:
                if self.improvements[-1] < self.thresh_upd:
                    self.count_under_thresh += 1
                else:
                    self.count_under_thresh = 0

                if self.verbose:
                    self._write("  Curr. impr.: {:.2E}\n".format(self.improvements[-1]))
                    self._write("  # times under thresh: {}\n".format(self.count_under_thresh))

                if self.count_under_thresh == self.count_upd:
                    self.count_under_thresh = 0

                    new_bs = int(self.factor_upd * batch)

                    if new_bs >= self.max:
                        new_batch = self.max
                        self.full_data = True
                    else:
                        new_batch = new_bs

                    if self.verbose:
                        self._write("  -> New batch size of {} samples\n".format(new_batch))

        return new_batch

    def _write(self, msg):
        sys.stderr.write(msg)


def wma(values, curr_iter, window_size):
    """
    Window Moving Average

    :param values: values to compute the wma on
    :param curr_iter: current iteration
    :param window_size: Size of the window
    :return: Value of the WMA
    """

    curr_iter = int(curr_iter)
    if curr_iter < window_size:
        window_size = curr_iter

    window_size = int(window_size)

    tmp = [(i + 1) * v for i, v in enumerate(values[curr_iter - window_size:curr_iter])]

    wma = np.sum(tmp) / np.sum(range(1, window_size + 1))

    return wma


def ema(values, curr_iter, window_size, decay=1):
    """
    Window Moving Average with exponential decay

    :param values: values to compute the wma on
    :param curr_iter: current iteration
    :param window_size: Size of the window
    :param decay: Value of the exponential decay
    :return: Value of the WMA
    """

    if curr_iter < window_size:
        window_size = curr_iter

    weights = np.exp(np.linspace(-decay, 0., window_size))
    weights /= weights.sum()

    tmp = [v * w for v, w in zip(values[curr_iter - window_size:curr_iter], weights)]

    ema = np.sum(tmp)

    return ema


