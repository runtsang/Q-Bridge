"""QuantumVariationalCircuit: Pennylane + Qiskit hybrid utilities.

This module exposes two utilities:

- `QiskitVariationalCircuit`: a two‑qubit parameterised circuit that can be run on
  a Qiskit Aer simulator.  It implements a simple Ry‑CX‑Ry ansatz and
  returns the expectation of Pauli‑Z on each qubit.

- `PennylaneVariationalQNode`: a Pennylane QNode that mirrors the
  architecture used in `QuantumNATHybridQ`.  It is fully differentiable
  with PyTorch and can be embedded as a layer inside a neural network.
"""

from __future__ import annotations

import numpy as np
import torch
import pennylane as qml
from pennylane import numpy as pnp

# --------------------------------------------------------------------------- #
# Qiskit implementation
# --------------------------------------------------------------------------- #
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import Aer

class QiskitVariationalCircuit:
    """Two‑qubit variational circuit with an Ry‑CX‑Ry ansatz.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (default 2).
    shots : int
        Number of measurement shots.
    """
    def __init__(self, n_qubits: int = 2, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("aer_simulator")

        self.circuit = QuantumCircuit(n_qubits)
        theta = qiskit.circuit.Parameter("theta")
        phi = qiskit.circuit.Parameter("phi")
        for q in range(n_qubits):
            self.circuit.ry(theta, q)
        self.circuit.cx(0, 1)
        for q in range(n_qubits):
            self.circuit.ry(phi, q)
        self.circuit.measure_all()

        self.theta = theta
        self.phi = phi

    def run(self, params: np.ndarray) -> np.ndarray:
        """Execute the circuit for the supplied parameters.

        Parameters
        ----------
        params : array_like
            Shape (2,) containing theta and phi.

        Returns
        -------
        expectations : ndarray
            Expectation values of Pauli‑Z for each qubit.
        """
        theta_val, phi_val = params
        compiled = transpile(self.circuit, self.backend)
        bound = compiled.bind_parameters({self.theta: theta_val, self.phi: phi_val})
        qobj = assemble(bound, shots=self.shots)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        expectations = []
        for qubit in range(self.n_qubits):
            exp = 0.0
            for bitstring, cnt in counts.items():
                # bitstring order is reversed in qiskit
                if bitstring[::-1][qubit] == "0":
                    exp += cnt
                else:
                    exp -= cnt
            expectations.append(exp / self.shots)
        return np.array(expectations)

# --------------------------------------------------------------------------- #
# Pennylane implementation
# --------------------------------------------------------------------------- #
class PennylaneVariationalQNode:
    """Pennylane QNode that implements a 4‑qubit ansatz.

    The circuit encodes the first four input features as Ry rotations,
    applies a parameterised RZ layer, and measures Pauli‑Z on each qubit.
    """
    def __init__(self, n_qubits: int = 4, dev_name: str = "default.qubit") -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device(dev_name, wires=n_qubits)
        self._qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Encode features
        for i in range(self.n_qubits):
            qml.RY(x[i], wires=i)
        # Parameterised layer
        for i in range(self.n_qubits):
            qml.RZ(params[i], wires=i)
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
        # Measure
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def __call__(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._qnode(params, x)

__all__ = ["QiskitVariationalCircuit", "PennylaneVariationalQNode"]
