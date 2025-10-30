"""
Quantum module that implements a hybrid estimator using a  
parameterised circuit, a quantum self‑attention block and a
quantum convolution filter.  Each component mirrors its classical
counterpart but operates on quantum states, enabling experiments
with full quantum‑classical integration.

The module exposes three reusable classes:
* QuantumConvFilter – a small n‑qubit circuit that implements a
  learnable filter via a random circuit and a threshold.
* QuantumSelfAttention – a block that applies rotation and
  entanglement gates to encode self‑attention logic.
* EstimatorQNNQuantum – a circuit that returns the expectation
  value of a Pauli‑Y observable.  The circuit can be driven by
  classical data (input angles) and trainable weights.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit.random import random_circuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Pauli


class QuantumConvFilter:
    """Quantum implementation of a convolution filter."""

    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 1024, threshold: float = 0.5):
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        self.circuit = QuantumCircuit(self.n_qubits)
        self.params = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i, p in enumerate(self.params):
            self.circuit.rx(p, i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Args:
            data: 2‑D array with shape (kernel_size, kernel_size)
        Returns:
            float: average probability of measuring |1> across qubits.
        """
        flat = data.reshape(1, self.n_qubits)
        param_binds = [{p: np.pi if val > self.threshold else 0 for p, val in zip(self.params, row)} for row in flat]
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        counts = 0
        for key, val in result.items():
            counts += val * sum(int(b) for b in key)
        return counts / (self.shots * self.n_qubits)


class QuantumSelfAttention:
    """Quantum self‑attention block."""

    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        self.rotation_params = [Parameter(f"rot{i}") for i in range(n_qubits * 3)]
        self.entanglement_params = [Parameter(f"ent{i}") for i in range(n_qubits - 1)]

        self.circuit = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            self.circuit.rx(self.rotation_params[3 * i], i)
            self.circuit.ry(self.rotation_params[3 * i + 1], i)
            self.circuit.rz(self.rotation_params[3 * i + 2], i)
        for i in range(n_qubits - 1):
            self.circuit.crx(self.entanglement_params[i], i, i + 1)
        self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, rotation_vals: np.ndarray, entangle_vals: np.ndarray, shots: int | None = None) -> np.ndarray:
        """
        Execute the circuit with the provided parameters.
        Returns the measurement outcome counts as a numpy array.
        """
        shots = shots or self.shots
        param_binds = {
            **{p: val for p, val in zip(self.rotation_params, rotation_vals)},
            **{p: val for p, val in zip(self.entanglement_params, entangle_vals)},
        }
        job = execute(self.circuit, self.backend, shots=shots, parameter_binds=[param_binds])
        result = job.result().get_counts(self.circuit)
        return np.array([int(k[::-1].count("1")) for k in result.keys()])


class EstimatorQNNQuantum:
    """Quantum expectation head for regression."""

    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        self.input_params = [Parameter(f"inp{i}") for i in range(n_qubits)]
        self.weight_params = [Parameter(f"wei{i}") for i in range(n_qubits - 1)]

        self.circuit = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            self.circuit.rx(self.input_params[i], i)
        for i in range(n_qubits - 1):
            self.circuit.crx(self.weight_params[i], i, i + 1)
        self.circuit.barrier()
        self.circuit.measure_all()

        # Pauli Y observable on the first qubit
        self.observable = Pauli.from_label("Y" + "I" * (n_qubits - 1))

    def expectation(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Args:
            inputs:  shape (batch, n_qubits)
            weights: shape (n_qubits - 1,)
        Returns:
            numpy array of expectation values, shape (batch,)
        """
        batch = inputs.shape[0]
        expectations = []
        for sample in inputs:
            param_bind = {p: val for p, val in zip(self.input_params, sample)}
            param_bind.update({p: val for p, val in zip(self.weight_params, weights)})
            job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[param_bind])
            result = job.result()
            counts = result.get_counts(self.circuit)
            # Convert counts to expectation of Y
            exp = 0.0
            for outcome, cnt in counts.items():
                bit = int(outcome[-1])  # first qubit
                val = 1 if bit == 0 else -1
                exp += val * cnt
            exp /= self.shots
            expectations.append(exp)
        return np.array(expectations)

    def forward(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Wrapper to allow use in PyTorch models.
        """
        inp_np = inputs.detach().cpu().numpy()
        wei_np = weights.detach().cpu().numpy()
        exp = self.expectation(inp_np, wei_np)
        return torch.tensor(exp, dtype=inputs.dtype, device=inputs.device)


__all__ = [
    "QuantumConvFilter",
    "QuantumSelfAttention",
    "EstimatorQNNQuantum",
]
