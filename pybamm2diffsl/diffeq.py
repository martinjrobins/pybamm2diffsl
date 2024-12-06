import ctypes
from functools import partial
from requests import Response
from wasmtime import Linker, Store, Module, WasiConfig
from numpy import ndarray
import numpy as np
from ctypes import c_ubyte

from .options import Options
from .solver import Solver
from .vector import Vector

import requests


class Diffeq:
    baseUrl = "https://diffeq-backend-staging.fly.dev"

    @classmethod
    def compile(cls, model: str) -> Response:
        url = cls.baseUrl + "/compile"
        data = {
            "text": model,
            "name": "unknown",
        }
        # body = json.dumps(data)
        # options = {
        #     "method": "POST",
        #     "mode": "cors",
        #     "headers": {
        #         "Content-Type": "application/json",
        #     },
        #     "body": body,
        # }
        r = requests.post(url, json=data, stream=True)
        r.raise_for_status()
        return r

    def __init__(self, model: str):
        self._model = model
        wasm_bytes = self.compile(model).content
        self._store = Store()
        wasi = WasiConfig()
        wasi.inherit_stdout()
        wasi.inherit_stderr()
        wasi.inherit_stdin()
        wasi.inherit_argv()
        wasi.inherit_env()
        self._store.set_wasi(wasi)
        linker = Linker(self._store.engine)
        linker.define_wasi()
        self._module = Module(self._store.engine, wasm_bytes)
        self._linking = linker.instantiate(self._store, self._module)

        exports = self._linking.exports(self._store)
        self.Solver_create = partial(exports["Sundials_create"], self._store)
        self.Solver_destroy = partial(exports["Sundials_destroy"], self._store)
        self.Solver_solve = partial(exports["Sundials_solve"], self._store)
        self.Solver_init = partial(exports["Sundials_init"], self._store)
        self.Solver_number_of_states = partial(
            exports["Sundials_number_of_states"], self._store
        )
        self.Solver_number_of_inputs = partial(
            exports["Sundials_number_of_inputs"], self._store
        )
        self.Solver_number_of_outputs = partial(
            exports["Sundials_number_of_outputs"], self._store
        )

        self.Options_create = partial(exports["Options_create"], self._store)
        self.Options_destroy = partial(exports["Options_destroy"], self._store)
        self.Options_set_fixed_times = partial(
            exports["Options_set_fixed_times"], self._store
        )
        self.Options_set_print_stats = partial(
            exports["Options_set_print_stats"], self._store
        )
        self.Options_set_fwd_sens = partial(
            exports["Options_set_fwd_sens"], self._store
        )
        self.Options_get_fixed_times = partial(
            exports["Options_get_fixed_times"], self._store
        )
        self.Options_get_print_stats = partial(
            exports["Options_get_print_stats"], self._store
        )
        self.Options_get_fwd_sens = partial(
            exports["Options_get_fwd_sens"], self._store
        )
        self.Options_get_linear_solver = partial(
            exports["Options_get_linear_solver"], self._store
        )
        self.Options_set_linear_solver = partial(
            exports["Options_set_linear_solver"], self._store
        )
        self.Options_get_preconditioner = partial(
            exports["Options_get_preconditioner"], self._store
        )
        self.Options_set_preconditioner = partial(
            exports["Options_set_preconditioner"], self._store
        )
        self.Options_get_jacobian = partial(
            exports["Options_get_jacobian"], self._store
        )
        self.Options_set_jacobian = partial(
            exports["Options_set_jacobian"], self._store
        )
        self.Options_set_atol = partial(exports["Options_set_atol"], self._store)
        self.Options_get_atol = partial(exports["Options_get_atol"], self._store)
        self.Options_set_rtol = partial(exports["Options_set_rtol"], self._store)
        self.Options_get_rtol = partial(exports["Options_get_rtol"], self._store)
        self.Options_set_linsol_max_iterations = partial(
            exports["Options_set_linsol_max_iterations"], self._store
        )
        self.Options_get_linsol_max_iterations = partial(
            exports["Options_get_linsol_max_iterations"], self._store
        )
        self.Options_get_debug = partial(exports["Options_get_debug"], self._store)
        self.Options_set_debug = partial(exports["Options_set_debug"], self._store)

        self.Vector_create = partial(exports["Vector_create"], self._store)
        self.Vector_destroy = partial(exports["Vector_destroy"], self._store)
        self.Vector_get = partial(exports["Vector_get"], self._store)
        self.Vector_get_length = partial(exports["Vector_get_length"], self._store)
        self.Vector_resize = partial(exports["Vector_resize"], self._store)
        self.Vector_get_data = partial(exports["Vector_get_data"], self._store)
        self.Vector_linspace_create = partial(
            exports["Vector_linspace_create"], self._store
        )
        self.Vector_create_with_capacity = partial(
            exports["Vector_create_with_capacity"], self._store
        )
        self.Vector_push = partial(exports["Vector_push"], self._store)

    def memory_ndarray(self) -> ndarray:
        mem = self._linking.exports(self._store)["memory"].data_ptr(self._store)
        size = self._linking.exports(self._store)["memory"].data_len(self._store)
        ptr_type = ctypes.c_ubyte * size
        buffer = ptr_type.from_address(ctypes.addressof(mem))
        return np.frombuffer(buffer, dtype=np.uint8)

    def memory_ptr(self) -> "ctypes._Pointer[c_ubyte]":
        return self._linking.exports(self._store)["memory"].data_ptr(self._store)

    def memory_size(self) -> int:
        return self._linking.exports(self._store)["memory"].data_len(self._store)

    def solver(self, options: Options) -> Solver:
        return Solver(self, options)

    def vector(self, array: list | ndarray) -> Vector:
        return Vector(self, array)

    def options(
        self,
        fixed_times=False,
        print_stats=False,
        fwd_sens=False,
        atol=1e-6,
        rtol=1e-6,
        linear_solver=Options.LinearSolver.LINEAR_SOLVER_DENSE,
        preconditioner=Options.Preconditioner.PRECON_NONE,
        jacobian=Options.Jacobian.DENSE_JACOBIAN,
        linsol_max_iterations=100,
        debug=False,
    ) -> Options:
        return Options(
            self,
            fixed_times,
            print_stats,
            fwd_sens,
            atol,
            rtol,
            linear_solver,
            preconditioner,
            jacobian,
            linsol_max_iterations,
            debug,
        )
