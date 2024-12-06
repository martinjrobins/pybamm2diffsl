import unittest

from pybamm2diffsl.diffeq import Diffeq
from pybamm2diffsl.options import Options
from tests.logistic import logistic


class TestOptions(unittest.TestCase):
    def setUp(self):
        self.diffeq = Diffeq(logistic)

    def test_create(self):
        o = self.diffeq.options(False, False, False)
        self.assertFalse(o.get_fixed_times())
        self.assertFalse(o.get_print_stats())
        self.assertFalse(o.get_fwd_sens())
        o.destroy()

        o = self.diffeq.options(True, True, True)
        self.assertTrue(o.get_fixed_times())
        self.assertTrue(o.get_print_stats())
        self.assertTrue(o.get_fwd_sens())
        o.destroy()

        o = self.diffeq.options(linear_solver=Options.LinearSolver.LINEAR_SOLVER_DENSE)
        self.assertEqual(
            o.get_linear_solver(), Options.LinearSolver.LINEAR_SOLVER_DENSE
        )
        o.destroy()

        o = self.diffeq.options(linear_solver=Options.LinearSolver.LINEAR_SOLVER_KLU)
        self.assertEqual(o.get_linear_solver(), Options.LinearSolver.LINEAR_SOLVER_KLU)
        o.destroy()

        o = self.diffeq.options(preconditioner=Options.Preconditioner.PRECON_NONE)
        self.assertEqual(o.get_preconditioner(), Options.Preconditioner.PRECON_NONE)
        o.destroy()

        o = self.diffeq.options(preconditioner=Options.Preconditioner.PRECON_LEFT)
        self.assertEqual(o.get_preconditioner(), Options.Preconditioner.PRECON_LEFT)
        o.destroy()

        o = self.diffeq.options(jacobian=Options.Jacobian.DENSE_JACOBIAN)
        self.assertEqual(o.get_jacobian(), Options.Jacobian.DENSE_JACOBIAN)
        o.destroy()

        o = self.diffeq.options(jacobian=Options.Jacobian.SPARSE_JACOBIAN)
        self.assertEqual(o.get_jacobian(), Options.Jacobian.SPARSE_JACOBIAN)
        o.destroy()

        o = self.diffeq.options(atol=1e-3)
        self.assertEqual(o.get_atol(), 1e-3)
        o.destroy()

        o = self.diffeq.options(rtol=1e-3)
        self.assertEqual(o.get_rtol(), 1e-3)
        o.destroy()

        o = self.diffeq.options(linsol_max_iterations=100)
        self.assertEqual(o.get_linsol_max_iterations(), 100)
        o.destroy()

        o = self.diffeq.options(debug=True)
        self.assertTrue(o.get_debug())
        o.destroy()
