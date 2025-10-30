"""Quantum regression model using a parameterised variational circuit.

The implementation is inspired by:
- `QuantumRegression.py` for data generation.
- `QuantumClassifierModel.py` for the variational ansatz.
- `ClassicalQuantumBinaryClassification.py` for the hybrid expectation head.

The circuit encodes the input features with `RX` rotations, applies a depth‑
controlled variational layer, and measures the expectation of `Z` on each qubit.
The resulting feature vector is fed into a classical linear head to produce a
scalar regression output.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, Pauli

# --------------------------------------------------------------------------- #
# Data utilities – identical to the classical version
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Same synthetic data generator as in the classical module."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset returning feature arrays and regression targets."""
    def __init__(self, samples: int, num_features: int):
        self.states, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Variational ansatz builder (from QuantumClassifierModel)
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, ParameterVector, ParameterVector, list[Pauli]]:
    """Construct a layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [Pauli("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, encoding, weights, observables

# --------------------------------------------------------------------------- #
# Quantum expectation wrapper
# --------------------------------------------------------------------------- #
class QuantumExpectationFunction(torch.autograd.Function):
    """Autograd wrapper that runs a parameterised circuit and returns Z‑expectations."""
    @staticmethod
    def forward(ctx,
                inputs: torch.Tensor,
                circuit: QuantumCircuit,
                encoding: ParameterVector,
                observables: list[Pauli],
                backend,
                shift: float = 0.0) -> torch.Tensor:  # type: ignore[override]
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.encoding = encoding
        ctx.observables = observables
        ctx.backend = backend

        batch_size = inputs.shape[0]
        results = []
        for i in range(batch_size):
            param_dict = {p: float(inputs[i, j]) for j, p in enumerate(encoding)}
            bound = circuit.bind_parameters(param_dict)
            result = execute(bound, backend).result()
            state = Statevector(result.get_statevector())
            exp_vals = torch.tensor([state.expectation_value(obs).real for obs in observables], dtype=torch.float32)
            results.append(exp_vals)
        return torch.stack(results)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Finite‑difference gradient via parameter‑shift (placeholder)
        return None, None, None, None, None, None, None

# --------------------------------------------------------------------------- #
# Main quantum regression model
# --------------------------------------------------------------------------- #
class QuantumRegressionModel(nn.Module):
    """Quantum regression model using a variational circuit and a classical head.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (equal to input feature dimensionality).
    depth : int
        Depth of the variational ansatz.
    """
    def __init__(self, num_qubits: int, depth: int = 3) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)
        self.backend = Aer.get_backend("statevector_simulator")
        self.head = nn.Linear(num_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype!= torch.float32:
            x = x.float()
        expectations = QuantumExpectationFunction.apply(x, self.circuit, self.encoding,
                                                       self.observables, self.backend,
                                                       shift=0.0)
        return self.head(expectations).squeeze(-1)

__all__ = ["RegressionDataset", "generate_superposition_data", "QuantumRegressionModel"]
