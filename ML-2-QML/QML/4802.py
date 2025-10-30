from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

class QuantumHybridSampler:
    """
    Parameterized quantum circuit that concatenates a 2‑qubit sampler block
    with a 4‑qubit self‑attention block.  Parameters are exposed via
    ParameterVector objects to facilitate binding during evaluation.
    """
    def __init__(self) -> None:
        # Sampler block parameters
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 4)

        # Attention block parameters
        self.rotation_params = ParameterVector("rot", 12)   # 3 per qubit
        self.entangle_params = ParameterVector("ent", 3)

        # Build the full circuit
        self._circuit = QuantumCircuit(6)
        # Sampler part
        qc_s = QuantumCircuit(2)
        qc_s.ry(self.input_params[0], 0)
        qc_s.ry(self.input_params[1], 1)
        qc_s.cx(0, 1)
        qc_s.ry(self.weight_params[0], 0)
        qc_s.ry(self.weight_params[1], 1)
        qc_s.cx(0, 1)
        qc_s.ry(self.weight_params[2], 0)
        qc_s.ry(self.weight_params[3], 1)
        self._circuit.append(qc_s, [0, 1])

        # Attention part
        qc_a = QuantumCircuit(4)
        for i in range(4):
            qc_a.rx(self.rotation_params[3*i], i)
            qc_a.ry(self.rotation_params[3*i+1], i)
            qc_a.rz(self.rotation_params[3*i+2], i)
        for i in range(3):
            qc_a.crx(self.entangle_params[i], i, i+1)
        qc_a.measure_all()
        self._circuit.append(qc_a, [2, 3, 4, 5])

        # Backend for sampling
        self.backend = Aer.get_backend("qasm_simulator")

    def bind_parameters(self, values: np.ndarray) -> QuantumCircuit:
        """
        Bind a complete parameter vector to the circuit.  The expected ordering
        is [input0, input1, weight0..3, rot0..11, ent0..2].
        """
        if len(values)!= 2 + 4 + 12 + 3:
            raise ValueError("Parameter vector length mismatch.")
        mapping = dict(zip(self._circuit.parameters, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def run(self, parameter_values: np.ndarray,
            shots: int = 1024) -> dict:
        """
        Execute the bound circuit and return measurement counts.
        """
        circ = self.bind_parameters(parameter_values)
        job = execute(circ, self.backend, shots=shots)
        return job.result().get_counts(circ)

    def sample(self, parameter_values: np.ndarray,
               num_samples: int = 1000) -> np.ndarray:
        """
        Produce raw sample draws from the circuit’s output distribution.
        """
        counts = self.run(parameter_values, shots=num_samples)
        samples = []
        for outcome, freq in counts.items():
            samples.extend([list(map(int, reversed(outcome)))] * freq)
        return np.array(samples)

    def expectation(self, observables: list,
                    parameter_sets: list) -> list:
        """
        Evaluate expectation values using the FastBaseEstimator pattern.
        Each observable is a callable accepting a Statevector and returning a
        complex expectation value.
        """
        results = []
        for params in parameter_sets:
            circ = self.bind_parameters(params)
            state = Statevector.from_instruction(circ)
            results.append([obs(state) for obs in observables])
        return results

__all__ = ["QuantumHybridSampler"]
