from .options import Options


class Solver:
    def __init__(self, diffeq, options: Options):
        self.options = options
        self.pointer = diffeq.Solver_create()
        diffeq.Solver_init(self.pointer, options.pointer)
        self.number_of_inputs = diffeq.Solver_number_of_inputs(self.pointer)
        self.number_of_outputs = diffeq.Solver_number_of_outputs(self.pointer)
        self.number_of_states = diffeq.Solver_number_of_states(self.pointer)
        self.dummy_vector = diffeq.vector([])
        self.diffeq = diffeq

    def destroy(self):
        self.diffeq.Solver_destroy(self.pointer)

    def solve(self, times, inputs, outputs):
        if len(inputs) != self.number_of_inputs:
            raise ValueError(
                f"Expected {self.number_of_inputs} inputs, got {len(inputs)}"
            )
        if len(times) < 2:
            raise ValueError("Times vector must have at least two elements")
        result = self.diffeq.Solver_solve(
            self.pointer,
            times.pointer,
            inputs.pointer,
            self.dummy_vector.pointer,
            outputs.pointer,
            self.dummy_vector.pointer,
        )
        if result != 0:
            raise ValueError("Solve failed")

    def solve_with_sensitivities(self, times, inputs, dinputs, outputs, doutputs):
        if len(inputs) != self.number_of_inputs:
            raise ValueError(
                f"Expected {self.number_of_inputs} inputs, got {len(inputs)}"
            )
        if len(inputs) != len(dinputs):
            raise ValueError(f"Expected {len(inputs)} dinputs, got {len(dinputs)}")
        if len(times) < 2:
            raise ValueError("Times vector must have at least two elements")
        result = self.diffeq.Solver_solve(
            self.pointer,
            times.pointer,
            inputs.pointer,
            dinputs.pointer,
            outputs.pointer,
            doutputs.pointer,
        )
        if result != 0:
            raise ValueError("Solve failed")
