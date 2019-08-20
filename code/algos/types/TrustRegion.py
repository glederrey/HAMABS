"""
Code greatly inspired by:
    https://github.com/optimizers/nlpy/blob/master/nlpy/krylov/pcg.py
"""
from .Type import Type

import numpy as np


class TrustRegion(Type):

    def __init__(self, **kwargs):
        Type.__init__(self, **kwargs)

        self.batch = int(self.full_size)
        self.stocha = False

        # Parameters specific to Trust Region
        self.iters = kwargs.get('iters', 2*self.n)
        self.reltol = kwargs.get('reltol', 1.0e-6)
        self.delta0 = kwargs.get('delta0', 5)

        self.eta1 = kwargs.get('eta1', 0.01)
        self.eta2 = kwargs.get('eta2', 0.9)

        # Other parameters
        self.precondition = kwargs.get('precondition', lambda v: v)
        self.thresh = kwargs.get('thresh', 1.0e-5)
        self.deltak = self.delta0
        self.deltaM = 9999

        # Formats to display info about subproblem
        self.hd_fmt = '     %-5s  %9s  %8s\n'
        self.header = self.hd_fmt % ('Iter', '<r,g>', 'curv')
        self.fmt = '     %-5d  %9.2e  %8.2e\n'

    def init_solve(self):
        self.deltak = self.delta0

    def update_xk(self, xk, fk, gk, Bk, f, fprime, dir, fs):

        zk = self.solve_subProblem(gk, Bk, self.deltak)

        top = fk - f(xk + zk)

        bottom = self.approx(np.zeros(len(xk)), fk, gk, Bk) - self.approx(zk, fk, gk, Bk)

        rho_k = top / bottom

        if self.verbose:
            self._write("  rho = {:.3f}\n".format(rho_k))
            self._write("  deltak = {:.2f}\n".format(self.deltak))
            self._write("  zk = {}\n".format(zk))
            self._write("  ||zk|| = {:.3E}\n".format(np.linalg.norm(zk)))
            self._write("  status: {}\n".format(self.status))

        # Acceptance of trial point
        xk_new = xk
        if rho_k > self.eta1:
            xk_new = xk + zk

        # Update trust region radius
        if rho_k < self.eta1:
            self.deltak = 0.25*np.linalg.norm(zk)
        elif rho_k < self.eta2:
            self.deltak = np.linalg.norm(zk)
        else:# rho_k > self.eta2
            self.deltak = min(2.0*self.deltak, self.deltaM)

        return xk_new

    def solve_subProblem(self, gk, Bk, deltak):
        """
        Solve the subproblem of the Trust Region algorithm using Steihaug-Toint CG

        :param fk: value of the objective function at xk
        :param gk: value of the gradient at xk
        :param Bk: value of the Hessian at xk
        :param deltak: Radius of the Trust Region
        :return: zk the step for iteration k
        """

        # Initialization
        r = gk.copy()           # Avoid overwriting gk
        z = np.zeros(self.n)
        z_norm = np.dot(z, z)
        y = self.precondition(r)
        ry = np.dot(r, y)
        self.status = None

        exitOptimal = False
        exitIter = False
        try:
            sqrtry = np.sqrt(ry)
        except:
            msg = 'Preconditioned residual = {.3f}\n'.format(ry)
            msg += 'Is preconditioner positive definite?'
            raise ValueError(msg)

        stopTol = max(self.thresh, self.reltol*sqrtry)

        # p is the preconditionel residual
        p = -y
        k = 0

        onBoundary = False
        infDescent = False

        if self.verbose:
            self._write("  Solving sub-problem:\n")
            self._write(self.header)
            self._write('    '+'-' * (len(self.header)-4) + '\n')

        # Run the algo for solving the sub-problem
        while not (exitOptimal or exitIter) and not onBoundary and not infDescent:

            k+=1
            Bp = np.dot(Bk, p)
            pBp = np.dot(p, Bp)

            if self.verbose:
                self._write(self.fmt % (k, ry, pBp))

            sigma = self.to_boundary(z, p, deltak)

            # CG steplength
            alpha = ry/pBp

            if pBp <= 0 or alpha > sigma:
                if self.verbose:
                    self._write("    Boundary reached!\n")
                z += sigma * p
                znorm2 = deltak**2
                self.status = "Boundary active"
                onBoundary = True
                continue

            # Next iterate
            z += alpha * p
            r += alpha * Bp
            y = self.precondition(r)
            ry_next = np.dot(r, y)
            beta = ry_next/ry
            p = -y + beta*p
            ry = ry_next
            try:
                sqrtry = np.sqrt(ry)
            except:
                msg = 'Preconditioned residual = {.3f}\n'.format(ry)
                msg += 'Is preconditioner positive definite?'
                raise ValueError(msg)

            z_norm = np.dot(z, z)

            # Exit criterion
            exitIter = k >= self.iters
            exitOptimal = sqrtry <= stopTol

        return z

    def to_boundary(self, z, p, delta):
        """
        With the vector 'x' and 'p' and a radius delta > 0,
        find sigma such that
            || z + sigma * p || = radius

        where ||.|| is the Euclidian norm.

        :param x: actual point
        :param p: direction
        :param delta: radius
        :return:
        """
        zp = np.dot(z,p)
        pp = np.dot(p,p)
        zz = np.dot(z,z)
        sigma = (-zp + np.sqrt(zp*zp + pp * (delta**2 - zz)))
        sigma /= pp
        return sigma

    def approx(self, x, f, g, H):
        """

        Return the approximation of the Taylor expansion:
            f(x) -> f + <g,x> + 1/2 <x, Hx>

        :param x: point
        :param f: value of the function at x0
        :param g: value of the gradient at x0
        :param H: value of the Hessian at x0
        :return: approximation of the function f at x
        """
        return f + np.dot(g, x) + 0.5 * np.dot(x, np.dot(H, x))

    def to_str(self):
        return "Trust Region"



