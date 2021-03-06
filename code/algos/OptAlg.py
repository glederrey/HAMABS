"""
Container for all Optimization algorithms that we're using
"""

import sys
import time
import numpy as np
from scipy.optimize.optimize import OptimizeResult

from .directions import *
from .types import *
from .helpers import back_to_bounds, sc_grad, sc_rel_grad


class OptAlg:

    def __init__(self, alg_type=None, direction=None):

        # Check that the type of IOA and the direction are OK
        possible_types = ['LS', 'LS-ABS', 'TR', 'TR-ABS']

        if alg_type not in possible_types:
            raise ValueError("Please, give a type of IOA amongst the possible types: " + ' '.join(possible_types))
        self.alg_type_str = alg_type

        possible_directions = ['grad', 'hess', 'bfgs', 'bfgs-inv', 'hybrid', 'hybrid-inv']

        if direction not in possible_directions:
            raise ValueError("Please, give a direction for the IOA amongst the possible directions: " + ' '.join(
                possible_directions))
        self.dir_str = direction

        if direction == 'grad' and 'TR' in alg_type:
            raise ValueError("Trust Region does not work with gradient as the direction.")

        if direction == 'hybrid' and 'ABS' not in alg_type:
            raise ValueError("The Hybrid direction has to be used with an ABS algorithm.")

        if direction == 'bfgs-inv' and 'TR' in alg_type:
            raise ValueError("The Inverse BFGS update has to be used with a LineSearch algorithm.")

        if direction == 'hybrid-inv' and alg_type != 'LS-ABS':
            raise ValueError("The Hybrid with inverse BFGS can only be used with LS-ABS algorithm.")

        # Initialize some parameters

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
        self.max_epochs = kwargs.get('max_epochs', 1000)
        self.thresh = kwargs.get('thresh', 1e-6)
        self.bounds = kwargs.get('bounds', None)
        self.verbose = kwargs.get('verbose', False)
        self.seed = kwargs.get('seed', -1)
        self.stop_scrit_str = kwargs.get('stop_crit', 'rel_grad')
        self.compute_final_hessian = kwargs.get('compute_final_hessian', False)

        # Formats to display info about subproblem
        self.hd_fmt = '     %-5s  %9s  %8s\n'
        self.header = self.hd_fmt % ('Iter', '<r,g>', 'curv')
        self.fmt = '     %-5d  %9.2e  %8.2e\n'

        # Function
        self.f = lambda x: self.biogeme.calculateLikelihood(x)

        # Fix the issue with the bounds. We don't want to have None.
        # We want to have -inf or +inf instead.

        if self.bounds is not None:
            new_bounds = []
            for b in self.bounds:
                if b[0] is None and b[1] is None:
                    new_bounds.append((-np.inf, np.inf))
                elif b[0] is None and b[1] is not None:
                    new_bounds.append((-np.inf, b[1]))
                elif b[0] is not None and b[1] is None:
                    new_bounds.append((b[0], np.inf))
                else:
                    new_bounds.append((b[0], b[1]))

            self.bounds = new_bounds
            kwargs['bounds'] = self.bounds

        # Prepare the direction
        if self.dir_str == 'grad':
            self.dir = Gradient(**kwargs)
        elif self.dir_str == 'hess':
            self.dir = Hessian(**kwargs)
        elif self.dir_str == 'bfgs':
            self.dir = BFGS(**kwargs)
        elif self.dir_str == 'bfgs-inv':
            self.dir = BFGS_INV(**kwargs)
        elif self.dir_str == 'hybrid':
            self.dir = Hybrid(**kwargs)
        elif self.dir_str == 'hybrid-inv':
            self.dir = Hybrid_INV(**kwargs)

        # Prepare the type
        if self.alg_type_str == 'LS':
            self.alg_type = LineSearch(**kwargs)
        elif self.alg_type_str == 'LS-ABS':
            self.alg_type = LineSearchABS(**kwargs)
        elif self.alg_type_str == 'TR':
            self.alg_type = TrustRegion(**kwargs)
        elif self.alg_type_str == 'TR-ABS':
            self.alg_type = TrustRegionABS(**kwargs)

        if self.stop_scrit_str == 'rel_grad':
            self.stop_crit = sc_rel_grad
        else:
            self.stop_crit = sc_grad

    def solve(self, maximize=False):
        """
        Minimize the objective function f starting from x0.

        :return: x: the optimized parameters
        """
        start_time = time.time()

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

        if self.seed != -1:
            np.random.seed(self.seed)

        self.ep = 0
        self.it = 0

        # Initialize the multiplicative factor for the direction
        self.dir.prep_mult_factor(maximize)

        # Initialize the algorithm
        self.alg_type.init_solve()

        # Initialize the Hessian to an identity matrix for BFGS
        # Nothing for the other directions
        Bk = self.dir.init_hessian(self.x0)

        while self.ep < self.max_epochs:

            # Get the function, its gradient and the Hessian
            f, fprime, grad_hess = self.dir.compute_func_and_derivatives(self.alg_type.batch, self.alg_type.full_size)

            fk = f(xk)
            gk, Bk = grad_hess(xk, Bk)

            # Add the return values to the arrays
            self.xs.append(xk)
            self.fs.append(fk)
            self.epochs.append(self.ep)
            self.batches.append(self.alg_type.batch)

            sc = self.stop_crit(xk, fk, gk)

            if 0 < sc <= self.thresh and self.alg_type.batch == self.alg_type.full_size:
                if self.verbose:
                    self._write("Algorithm Optimized!\n")
                    self._write("  x* = [{}]\n".format(", ".join(format(x, ".3f") for x in xk)))
                    self._write("  f(x*) = {:.3f}\n".format(fk))

                self.optimized = True
                break

            if self.verbose:
                self._write("Epoch {}:\n".format(self.ep))
                self._write("  xk = [{}]\n".format(", ".join(format(x, ".3f") for x in xk)))
                self._write("  f(xk) = {:.3f}\n".format(fk))
                self._write("  ||gk|| = {:.3E}\n".format(np.linalg.norm(gk)))
                self._write("  stop_crit = {:.3E}\n".format(sc))

            # Get the new value for x_k using either a LineSearch or a TrustRegion algorithm
            xk_new = self.alg_type.update_xk(xk, fk, gk, Bk, f, fprime, self.dir, self.fs)

            # Update the hessian; only used by BFGS.
            # Return the same Bk for the other directions
            Bk = self.dir.upd_hessian(xk, xk_new, f, fprime, Bk, gk)

            # Make sure xk is still in bounds
            xk = back_to_bounds(xk_new, self.bounds)

            # Update the batch size if we're using an ABS algorithm
            self.dir.batch_changed = self.alg_type.update_batch(self.it, fk/self.alg_type.batch)

            # Update the Hybrid direction if needed (does nothing for all other directions)
            self.dir.update_dir(self.alg_type.batch, self.alg_type.full_size)

            if self.verbose:
                self._write('\n')

            # Update the number of epochs
            self.ep += self.alg_type.batch/self.alg_type.full_size
            self.it += 1

        status = 'Algorithm optimized'
        if not self.optimized:
            status = 'Optimum not reached'

        if self.verbose and not self.optimized:

            self._write("Algorithm not fully optimized!\n")
            self._write("  x_n = [{}]\n".format(", ".join(format(x, ".3f") for x in xk)))
            self._write("  f(x_n) = {:.3f}\n".format(fk))

        self.opti_time = time.time() - start_time

        # Compute the function value, the gradient and the Hessian one last time.
        fk, gk, Bk = self.dir.compute_final_LL_and_derivatives(xk, hessian=self.compute_final_hessian)

        dct = {'x': xk,
               'success': self.optimized,
               'status': status,
               'fun': fk,
               'jac': gk,
               'nit': self.it,
               'nep': self.ep,
               'stop_crit': sc,
               'opti_time': self.opti_time}

        if self.compute_final_hessian:
            dct['hess'] = Bk

        return OptimizeResult(dct)

    def _write(self, msg):
        sys.stderr.write(msg)
