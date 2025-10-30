"""Quantum regression QCNN model derived from the QCNN and quantum regression seeds."""
from __future__ import annotations

import numpy as np
import torch
import torch.quantum as tq  # placeholder for a quantum‑tensor library
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from torch.utils.data import Dataset


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate states of the form cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩.
    Labels are sin(2θ)cos(φ), providing a smooth regression target.
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset for quantum regression, yielding complex state vectors."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {"states": torch.tensor(self.states[index], dtype=torch.cfloat),
                "target": torch.tensor(self.labels[index], dtype=torch.float32)}


class QCNNRegressionQNN:
    """
    Quantum CNN ansatz for regression.  Builds a QCNN circuit with a Z‑feature
    map, trainable convolution and pooling layers, and a single‑qubit Z
    observable.  The model is evaluated via a state‑vector estimator.
    """
    def __init__(self, num_wires: int = 8):
        self.num_wires = num_wires
        self.estimator = Estimator()
        self.feature_map = ZFeatureMap(num_wires)
        self.circuit = self._build_ansatz()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_wires - 1), 1)])
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Single two‑qubit convolution block."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        """Stack of convolution blocks over adjacent qubit pairs."""
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        params = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            block = self._conv_circuit(params[i * 3 : (i + 2) * 3])
            qc.append(block, [i, i + 1])
            qc.barrier()
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Two‑qubit pooling block."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(self, sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
        """Pooling over pairs of source‑sink qubits."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        params = ParameterVector(prefix, length=len(sources) * 3)
        for src, sink in zip(sources, sinks):
            block = self._pool_circuit(params[:3])
            qc.append(block, [src, sink])
            qc.barrier()
            params = params[3:]
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        """Assemble the full QCNN ansatz."""
        ansatz = QuantumCircuit(self.num_wires, name="Ansatz")

        # First convolution + pooling
        ansatz.compose(self._conv_layer(self.num_wires, "c1"), inplace=True)
        ansatz.compose(self._pool_layer(list(range(self.num_wires // 2)),
                                       list(range(self.num_wires // 2, self.num_wires)),
                                       "p1"), inplace=True)

        # Second convolution + pooling
        ansatz.compose(self._conv_layer(self.num_wires // 2, "c2"), inplace=True)
        ansatz.compose(self._pool_layer(list(range(self.num_wires // 4)),
                                       list(range(self.num_wires // 4, self.num_wires // 2)),
                                       "p2"), inplace=True)

        # Third convolution + pooling (single qubit)
        ansatz.compose(self._conv_layer(self.num_wires // 4, "c3"), inplace=True)
        ansatz.compose(self._pool_layer([0], [1], "p3"), inplace=True)

        return ansatz

    def __call__(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Evaluate the QCNN on a batch of input states."""
        return self.qnn(input_batch).squeeze(-1)


def QCNNRegressionQNNFactory() -> QCNNRegressionQNN:
    """Convenience factory returning a fully‑initialized QCNN regression QNN."""
    return QCNNRegressionQNN(num_wires=8)


__all__ = ["QCNNRegressionQNN", "QCNNRegressionQNNFactory", "RegressionDataset", "generate_superposition_data"]
