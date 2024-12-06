import unittest
import numpy as np
import pybamm
import requests

from pybamm2diffsl.diffeq import Diffeq
from pybamm2diffsl.options import Options
from pybamm2diffsl.pybamm_model import PybammModel


class TestConvert(unittest.TestCase):
    def checks(
        self,
        model: str,
        test_name: str,
        inputs: list,
        outputs: list,
        pybamm_model: PybammModel,
    ):
        try:
            diffeq = Diffeq(model)
        except requests.exceptions.HTTPError as e:
            inpts_str = "_".join(inputs)
            outputs_str = "_".join(outputs)
            filename = f"{test_name}_{inpts_str}_{outputs_str}"
            with open(f"tests/{filename}.ds", "w") as f:
                f.write(model)
            with open(f"tests/{filename}.txt", "w") as f:
                f.write(e.response.text)
            self.fail(f"{filename} failed with: {e.response.text}")

        end_time = 3600
        output = outputs[0]

        # solve model with pybamm
        pv = pybamm_model.model.default_parameter_values
        for inpt in inputs:
            pv[inpt] = "[input]"
        input_values = [0.5] * len(inputs)
        input_dict = dict(zip(inputs, input_values))

        sim = pybamm.Simulation(pybamm_model.model, parameter_values=pv)
        solve_times = np.linspace(0, end_time, 100)
        sol = sim.solve(solve_times, inputs=input_dict)
        pybamm_result = sol[output](solve_times)

        # solve with diffeq
        o = diffeq.options(
            atol=1e-6,
            rtol=1e-6,
            fixed_times=True,
            jacobian=Options.Jacobian.SPARSE_JACOBIAN,
            linear_solver=Options.LinearSolver.LINEAR_SOLVER_KLU,
            debug=True,
            print_stats=True,
        )
        s = diffeq.solver(o)
        times = diffeq.vector(solve_times)
        print("times len", len(times))
        diffeq_inputs = diffeq.vector(input_values)
        diffeq_outputs = diffeq.vector([])
        s.solve(times, diffeq_inputs, diffeq_outputs)
        take_every = len(outputs) - 1
        if take_every == 0:
            diffeq_result = diffeq_outputs.getFloat64Array()
        else:
            diffeq_result = diffeq_outputs.getFloat64Array()[::take_every]
        print("times len", len(times))
        try:
            np.testing.assert_array_almost_equal(pybamm_result, diffeq_result)
        except AssertionError as e:
            inpts_str = "_".join(inputs)
            outputs_str = "_".join(outputs)
            filename = f"{test_name}_{inpts_str}_{outputs_str}"
            with open(f"tests/{filename}.ds", "w") as f:
                f.write(model)
            with open(f"tests/{filename}_pybamm.txt", "w") as f:
                f.write(str(pybamm_result))
            with open(f"tests/{filename}_diffeq.txt", "w") as f:
                f.write(str(diffeq_result))
            self.fail(f"{filename} failed with: {e}")
        s.destroy()
        o.destroy()

    def test_spm_no_inputs(self):
        model = PybammModel(pybamm.lithium_ion.SPM())
        output = "Voltage [V]"
        result = model.to_diffeq(inputs=[], outputs=[output])
        self.checks(result, "test_spm_no_inputs", [], [output], model)

    def test_spm_current_input_voltage_output(self):
        model = PybammModel(pybamm.lithium_ion.SPM())
        inpt = "Current function [A]"
        output = "Voltage [V]"
        result = model.to_diffeq(inputs=[inpt], outputs=[output])
        self.checks(
            result, "test_spm_current_input_voltage_output", [inpt], [output], model
        )

    def test_spm_all_inputs(self):
        model = PybammModel(pybamm.lithium_ion.SPM())
        all_inputs = model.get_all_parameters()
        for inpt in all_inputs:
            output = "Voltage [V]"
            result = model.to_diffeq(inputs=[inpt], outputs=[output])
            self.checks(result, "test_spm_all_inputs", [inpt], [output], model)

    def test_spm_all_outputs(self):
        model = PybammModel(pybamm.lithium_ion.SPM())
        all_outputs = model.get_all_outputs()
        for output in all_outputs:
            inpt = "Current function [A]"
            result = model.to_diffeq(inputs=[inpt], outputs=[output])
            self.checks(result, "test_spm_all_outputs", [inpt], [output], model)
