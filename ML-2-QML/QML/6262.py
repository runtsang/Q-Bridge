"""Quantum layer used by HybridEstimatorQNN. Implements a single‑qubit
variational circuit whose parameters are supplied by a classical tensor.
The layer returns the expectation value of the Pauli‑Z operator, which
is differentiable via the parameter‑shift rule when used with a
Qiskit statevector simulator.
"""

import numpy as np
import torch
from qiskit import QuantumCircuit, Aer, execute

class QuantumLayer:
    """
    Parameterized quantum circuit that maps a single real value to the
    expectation of Z.  The circuit consists of an H gate, followed by
    a rotation about Y (parameter theta) and a rotation about Z (parameter w).
    The expectation value of Z is returned.
    """
    def __init__(self, shots: int = 1024):
        self.shots = shots
        self.backend = Aer.get_backend("statevector_simulator")

        # Build the template circuit
        self.circuit = QuantumCircuit(1)
        self.theta = self.circuit.parameter("theta")
        self.w = self.circuit.parameter("w")
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.rz(self.w, 0)
        # No measurement; we use the statevector for expectation

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute the expectation of Z for each input value.

        Parameters
        ----------
        inputs : np.ndarray
            1‑D array of real values that will be bound to both theta and w.

        Returns
        -------
        np.ndarray
            1‑D array of expectation values.
        """
        flat = inputs.ravel()
        expectations = []
        for val in flat:
            bound = self.circuit.bind_parameters({self.theta: val, self.w: val})
            statevector = execute(bound, self.backend).result().get_statevector()
            # expectation of Z = |0>^2 - |1>^2
            exp_z = np.real(statevector[0] * np.conj(statevector[0]) -
                            statevector[1] * np.conj(statevector[1]))
            expectations.append(exp_z)
        return np.array(expectations, dtype=np.float32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Wrapper that accepts a torch tensor and returns a torch tensor.
        """
        with torch.no_grad():
            out = self.__call__(inputs.detach().cpu().numpy())
        return torch.tensor(out, dtype=torch.float32, device=inputs.device)

__all__ = ["QuantumLayer"]
