"""Hybrid quantum circuit combining a fully connected layer and a sampler network."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter, ParameterVector


class HybridQuantumCircuit:
    """
    Quantum circuit that implements both a fully connected layer (via a single rotation)
    and a sampler network (via a 2‑qubit parameterized sub‑circuit). The `run` method
    accepts a dictionary with keys:
        - 'fc_theta': scalar rotation for the fully connected part.
        -'sampler_inputs': 2‑dim input parameters for the sampler.
        -'sampler_weights': 4‑dim weight parameters for the sampler.
    It returns a NumPy array containing the expectation of qubit 0
    and a 2‑element probability vector derived from qubits 0 and 1.
    """

    def __init__(self, n_qubits: int = 3, backend=None, shots: int = 1024) -> None:
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Parameters
        self.fc_theta = Parameter("fc_theta")
        self.sampler_inputs = ParameterVector("input", 2)
        self.sampler_weights = ParameterVector("weight", 4)

        # Build circuit
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        qc.barrier()

        # Fully connected part (qubit 0)
        qc.ry(self.fc_theta, 0)

        # Sampler part (qubits 0 & 1)
        qc.ry(self.sampler_inputs[0], 0)
        qc.ry(self.sampler_inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(self.sampler_weights[0], 0)
        qc.ry(self.sampler_weights[1], 1)
        qc.cx(0, 1)
        qc.ry(self.sampler_weights[2], 0)
        qc.ry(self.sampler_weights[3], 1)

        qc.measure_all()
        self.circuit = qc

    def run(self, params: dict[str, Iterable[float]]) -> np.ndarray:
        # Prepare parameter bindings
        bind_list = [
            {self.fc_theta: params["fc_theta"][0]},
            {self.sampler_inputs[0]: params["sampler_inputs"][0]},
            {self.sampler_inputs[1]: params["sampler_inputs"][1]},
            {self.sampler_weights[0]: params["sampler_weights"][0]},
            {self.sampler_weights[1]: params["sampler_weights"][1]},
            {self.sampler_weights[2]: params["sampler_weights"][2]},
            {self.sampler_weights[3]: params["sampler_weights"][3]},
        ]
        bound_circuit = self.circuit.bind_parameters(bind_list)

        job = execute(bound_circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_circuit)

        # Convert counts to probabilities
        probs = {state: cnt / self.shots for state, cnt in counts.items()}

        # Expectation of qubit 0
        exp_fc = 0.0
        for state, p in probs.items():
            # State string like '010'
            exp_fc += int(state[-1]) * p  # qubit 0 is the last bit in Qiskit's ordering
        # Sampler distribution from qubits 0 & 1
        sampler_probs = np.zeros(2)
        for state, p in probs.items():
            q0 = int(state[-1])  # qubit 0
            q1 = int(state[-2])  # qubit 1
            if q0 == 0:
                sampler_probs[0] += p
            else:
                sampler_probs[1] += p
        return np.array([exp_fc, *sampler_probs])


__all__ = ["HybridQuantumCircuit"]
