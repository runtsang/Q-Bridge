"""Quantum estimator and kernel implementation.

This module builds on the EstimatorQNN example and QuantumKernelMethod
to provide a fully quantum EstimatorQNNGen345 class that can be used
in place of the classical surrogate.  It exposes a fit/predict
interface and a kernel_matrix routine based on TorchQuantum.
"""

import numpy as np
import torch
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from torchquantum.functional import func_name_dict, op_name_dict

# -------------------------------------------------------------
# Quantum kernel definitions
# -------------------------------------------------------------
class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""

    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = (
                x[:, info["input_idx"]]
                if op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = (
                -y[:, info["input_idx"]]
                if op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

# -------------------------------------------------------------
# Quantum Estimator
# -------------------------------------------------------------
class EstimatorQNNGen345:
    """Quantum surrogate estimator with a variational circuit.

    The circuit consists of a Hadamard layer, a data‑encoding Ry gate,
    and a variational Ry gate.  The estimator returns the expectation
    value of a Y observable on the first qubit.
    """

    def __init__(self, n_qubits: int = 4):
        # Parameter names
        self.input_params = [Parameter(f"x{i}") for i in range(n_qubits)]
        self.weight_params = [Parameter(f"w{i}") for i in range(n_qubits)]

        # Build circuit
        self.circuit = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            self.circuit.h(i)
            self.circuit.ry(self.input_params[i], i)
            self.circuit.ry(self.weight_params[i], i)

        # Observable
        self.observable = SparsePauliOp.from_list([("Y" + "I" * (n_qubits - 1), 1)])

        # Statevector estimator
        self.estimator = StatevectorEstimator()

    def _prepare_circuit(self, x: np.ndarray) -> QuantumCircuit:
        """Return a circuit with data encoded into input parameters."""
        qc = self.circuit.copy()
        for i, val in enumerate(x):
            qc.set_parameter(self.input_params[i], val)
        return qc

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return expectation values for each sample."""
        preds = []
        for x in X:
            qc = self._prepare_circuit(x)
            result = self.estimator.run(circuit=qc, observables=[self.observable]).result()
            exp_val = result.estimator_value[0]
            preds.append(exp_val.real)
        return np.array(preds)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 30, lr: float = 0.1) -> None:
        """Train the variational weights using a naive gradient loop.

        This is a placeholder: in practice one would use a quantum
        circuit gradient estimator such as Parameter Shift or AD.
        """
        # Reinitialize weight params
        self.weight_params = [Parameter(f"w{i}") for i in range(self.circuit.num_qubits)]
        self.circuit = QuantumCircuit(self.circuit.num_qubits)
        for i in range(self.circuit.num_qubits):
            self.circuit.h(i)
            self.circuit.ry(self.input_params[i], i)
            self.circuit.ry(self.weight_params[i], i)

        # Simple loop over epochs
        for epoch in range(epochs):
            loss = 0.0
            for x, target in zip(X, y):
                qc = self._prepare_circuit(x)
                result = self.estimator.run(circuit=qc, observables=[self.observable]).result()
                pred = result.estimator_value[0].real
                loss += (pred - target) ** 2
            loss /= len(X)
            if epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} loss: {loss:.4f}")

    def kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute Gram matrix using the quantum kernel."""
        kernel = QuantumKernel()
        return np.array(
            [
                [
                    kernel(
                        torch.tensor(x, dtype=torch.float32),
                        torch.tensor(y, dtype=torch.float32),
                    ).item()
                    for y in Y
                ]
                for x in X
            ]
        )

__all__ = ["EstimatorQNNGen345", "QuantumKernel"]
