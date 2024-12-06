from enum import Enum


class Options:
    class LinearSolver(Enum):
        LINEAR_SOLVER_DENSE = 0
        LINEAR_SOLVER_KLU = 1
        LINEAR_SOLVER_SPBCGS = 2
        LINEAR_SOLVER_SPFGMR = 3
        LINEAR_SOLVER_SPGMR = 4
        LINEAR_SOLVER_SPTFQMR = 5

    class Preconditioner(Enum):
        PRECON_NONE = 0
        PRECON_LEFT = 1
        PRECON_RIGHT = 2

    class Jacobian(Enum):
        DENSE_JACOBIAN = 0
        SPARSE_JACOBIAN = 1
        MATRIX_FREE_JACOBIAN = 2
        NO_JACOBIAN = 3

    def __init__(
        self,
        diffeq,
        fixed_times=False,
        print_stats=False,
        fwd_sens=False,
        atol=1e-6,
        rtol=1e-6,
        linear_solver=LinearSolver.LINEAR_SOLVER_DENSE,
        preconditioner=Preconditioner.PRECON_NONE,
        jacobian=Jacobian.DENSE_JACOBIAN,
        linsol_max_iterations=100,
        debug: bool = False,
    ):
        self.pointer = diffeq.Options_create()
        diffeq.Options_set_fixed_times(self.pointer, 1 if fixed_times else 0)
        diffeq.Options_set_print_stats(self.pointer, 1 if print_stats else 0)
        diffeq.Options_set_fwd_sens(self.pointer, 1 if fwd_sens else 0)
        diffeq.Options_set_atol(self.pointer, atol)
        diffeq.Options_set_rtol(self.pointer, rtol)
        diffeq.Options_set_linear_solver(self.pointer, linear_solver.value)
        diffeq.Options_set_preconditioner(self.pointer, preconditioner.value)
        diffeq.Options_set_jacobian(self.pointer, jacobian.value)
        diffeq.Options_set_linsol_max_iterations(self.pointer, linsol_max_iterations)
        diffeq.Options_set_debug(self.pointer, 1 if debug else 0)

        self.diffeq = diffeq

    def destroy(self):
        self.diffeq.Options_destroy(self.pointer)

    def get_fixed_times(self) -> bool:
        return self.diffeq.Options_get_fixed_times(self.pointer) == 1

    def get_print_stats(self) -> bool:
        return self.diffeq.Options_get_print_stats(self.pointer) == 1

    def get_fwd_sens(self) -> bool:
        return self.diffeq.Options_get_fwd_sens(self.pointer) == 1

    def get_atol(self) -> float:
        return self.diffeq.Options_get_atol(self.pointer)

    def get_rtol(self) -> float:
        return self.diffeq.Options_get_rtol(self.pointer)

    def get_linear_solver(self) -> LinearSolver:
        return Options.LinearSolver(self.diffeq.Options_get_linear_solver(self.pointer))

    def get_preconditioner(self) -> Preconditioner:
        return Options.Preconditioner(
            self.diffeq.Options_get_preconditioner(self.pointer)
        )

    def get_jacobian(self) -> Jacobian:
        return Options.Jacobian(self.diffeq.Options_get_jacobian(self.pointer))

    def get_linsol_max_iterations(self) -> int:
        return self.diffeq.Options_get_linsol_max_iterations(self.pointer)

    def get_debug(self) -> bool:
        return self.diffeq.Options_get_debug(self.pointer) == 1
