"""
Container for all Optimization algorithms that we're using
"""

import sys
import time
import numpy as np
from scipy.optimize.optimize import OptimizeResult

from .directions import *
from .types import *
from .helpers import back_to_bounds


class OptAlg:

    def __init__(self, alg_type=None, direction=None):

        # Check that the type of IOA and the direction are OK
        possible_types = ['LS', 'LS-ABS', 'TR', 'TR-ABS']

        if alg_type not in possible_types:
            raise ValueError("Please, give a type of IOA amongst the possible types: " + ' '.join(possible_types))
        self.alg_type_str = alg_type

        possible_directions = ['grad', 'hess', 'bfgs', 'hybrid']

        if direction not in possible_directions:
            raise ValueError("Please, give a direction for the IOA amongst the possible directions: " + ' '.join(
                possible_directions))
        self.dir_str = direction

        if direction == 'grad' and 'TR' in alg_type:
            raise ValueError("Trust Region does not work with gradient as the direction.")

        # Initalize some parameters

        # Return values
        self.xs = []
        self.epochs = []
        self.fs = []
        self.fs_full = []
        self.batches = []

        # Other variables
        self.f = None
        self.x = None
        self.ep = None
        self.it = None
        self.optimized = None
        self.opti_time = None

        # Multiplicative factor for maximizing/minimizing
        self.mult = 1

    def __prep__(self, x0, biogeme, **kwargs):

        # Main variables
        self.x0 = x0
        self.biogeme = biogeme

        # Add more info using Biogeme
        kwargs['full_size'] = len(self.biogeme.database.data)
        kwargs['biogeme'] = self.biogeme
        kwargs['x0'] = self.x0

        # Some variables used for the algorithm
        self.status = '?'
        self.n = len(x0)

        # Other parameters in kwargs
        self.nbr_epochs = kwargs.get('nbr_epochs', 20)
        self.thresh = kwargs.get('thresh', 1.0e-5)
        self.bounds = kwargs.get('bounds', None)
        self.verbose = kwargs.get('verbose', False)
        self.seed = kwargs.get('seed', -1)

        # Formats to display info about subproblem
        self.hd_fmt = '     %-5s  %9s  %8s\n'
        self.header = self.hd_fmt % ('Iter', '<r,g>', 'curv')
        self.fmt = '     %-5d  %9.2e  %8.2e\n'

        # Function
        self.f = lambda x: self.biogeme.calculateLikelihood(x)

        # Prepare the direction
        if self.dir_str == 'grad':
            self.dir = Gradient(**kwargs)
            # We need a big threshold. Otherwise, there are some issues. =(
            self.thresh = 1e-4
        elif self.dir_str == 'hess':
            self.dir = Hessian(**kwargs)
        elif self.dir_str == 'bfgs':
            self.dir = BFGS(**kwargs)

        # Prepare the type
        if self.alg_type_str == 'LS':
            self.alg_type = LineSearch(**kwargs)
        elif self.alg_type_str == 'LS-ABS':
            self.alg_type = LineSearchABS(**kwargs)

        print(self.alg_type)

    def solve(self, maximize=False):
        """
        Minimize the objective function f starting from x0.

        :return: x: the optimized parameters
        """
        start_time = time.clock()

        if self.verbose:
            if maximize:
                solving = 'Maximizing'
            else:
                solving = 'Minimizing'

            self._write("{} the problem using a {} IOA with the {} direction.\n".format(solving,
                                                                                        self.alg_type.to_str(),
                                                                                        self.dir.to_str()))

        xk = np.array(self.x0)
        self.optimized = False

        # Return values set to Nothing
        self.xs = []
        self.epochs = []
        self.fs = []
        self.fs_full = []
        self.batches = []

        if maximize:
            self.mult = -1
        else:
            self.mult = 1

        if self.seed != -1:
            np.random.seed(self.seed)

        self.ep = 0
        self.it = 0

        # Initialize the Hessian to an identity matrix for BFGS
        # Nothing for the other directions
        Bk = self.dir.init_hessian(self.x0)

        while self.ep < self.nbr_epochs:

            # Compute the value of the obj function on all of the data
            fk_full = self.dir.compute_full_f(xk)

            # Get the function, its gradient and the Hessian
            f, fprime, grad_hess = self.dir.compute_func_and_derivatives(self.mult, self.alg_type.batch,
                                                                         self.alg_type.full_size)

            fk = f(xk)
            gk, Bk = grad_hess(xk, Bk)

            if np.linalg.norm(gk) <= self.thresh:
                if self.verbose:
                    self._write("Algorithm Optimized!\n")
                    self._write("  x* = [{}]\n".format(", ".join(format(x, ".3f") for x in xk)))
                    self._write("  f(x*) = {:.3f}\n".format(fk))

                self.optimized = True
                break

            # Add the return values to the arrays
            self.xs.append(xk)
            self.fs.append(fk)
            self.fs_full.append(fk_full)
            self.epochs.append(self.ep)
            self.batches.append(self.alg_type.batch)

            if self.verbose:
                self._write("Epoch {}:\n".format(self.ep))
                self._write("  xk = [{}]\n".format(", ".join(format(x, ".3f") for x in xk)))
                self._write("  f(xk) = {:.3f}\n".format(fk_full))

            # Compute the direction
            direction = self.dir.compute_direction(xk, gk, Bk)

            # Compute the value for alpha
            alpha = self.alg_type.compute_alpha(f, fprime, xk, direction, self.fs)

            if self.verbose:
                self._write("  ||gk|| = {:.3E}\n".format(np.linalg.norm(gk)))
                self._write("  ||dir|| = {:.3E}\n".format(np.linalg.norm(direction)))
                self._write("  alpha = {:.3E}\n".format(alpha))

            # Update the parameter value
            xk_new = xk + alpha * direction

            # Update the hessian; only used by BFGS.
            # Return the same Bk for the other directions
            Bk = self.dir.upd_hessian(xk, xk_new, f, fprime, Bk)

            # Make sure xk is still in bounds
            xk = back_to_bounds(xk_new, self.bounds)

            # Update the batch size if we're using an ABS algorithm
            self.alg_type.update_batch(self.it, fk_full)

            if self.verbose:
                self._write('\n')

            # Update the number of epochs
            self.ep += self.alg_type.batch/self.alg_type.full_size
            self.it += 1

        status = 'Algorithm optimized'
        if not self.optimized:
            status = 'Optimum not reached'

        if self.verbose and not self.optimized:
            fk_full = self.dir.compute_full_f(xk)

            f, fprime, grad_hess = self.dir.compute_func_and_derivatives(self.mult, self.alg_type.batch, self.alg_type.full_size)

            gk, Bk = grad_hess(xk, Bk)

            self._write("Algorithm not fully optimized!\n")
            self._write("  x_n = [{}]\n".format(", ".join(format(x, ".3f") for x in xk)))
            self._write("  f(x_n) = {:.3f}\n".format(fk_full))

        self.opti_time = time.clock() - start_time

        dct = {'x': xk,
               'success': self.optimized,
               'status': status,
               'fun': fk_full,
               'jac': gk,
               'hess': Bk,
               'nit': self.it,
               'nep': self.ep,
               'opti_time': self.opti_time}

        return OptimizeResult(dct)

    def _write(self, msg):
        sys.stderr.write(msg)
