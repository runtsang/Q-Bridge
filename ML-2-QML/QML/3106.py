import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 12345

class QCNNRegressionHybrid(nn.Module):
    """
    Quantum hybrid model that implements a QCNN followed by a classical regression head.
    The circuit consists of a ZFeatureMap, multiple convolutional and pooling layers
    built from a parameterised two‑qubit block, and a final measurement of the first
    qubit.  The expectation value is passed through a linear layer to produce a
    continuous output.
    """
    def __init__(self, num_wires: int, num_features: int):
        super().__init__()
        self.num_wires = num_wires
        self.num_features = num_features

        # Feature map
        self.feature_map = ZFeatureMap(num_features)

        # Ansatz
        self.ansatz = self._build_ansatz()

        # Combine feature map and ansatz
        circuit = QuantumCircuit(num_wires)
        circuit.compose(self.feature_map, range(num_wires), inplace=True)
        circuit.compose(self.ansatz, range(num_wires), inplace=True)

        # Observable: measure Z on the first qubit
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_wires - 1), 1)])

        # Estimator
        estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=estimator
        )

        # Classical regression head
        self.head = nn.Linear(1, 1)

    def _single_block(self, params: ParameterVector) -> QuantumCircuit:
        """
        Two‑qubit variational block used in convolution and pooling layers.
        """
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

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        """
        Convolutional layer that applies the single block to neighboring qubits.
        """
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            block = self._single_block(params[i * 3:(i + 2) * 3])
            qc.compose(block, [i, i + 1], inplace=True)
            qc.barrier()
        return qc

    def _pool_layer(self, sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
        """
        Pooling layer that condenses information from source qubits into sink qubits.
        """
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        params = ParameterVector(param_prefix, length=len(sources) * 3)
        for src, snk, p in zip(sources, sinks, range(len(sources))):
            block = self._single_block(params[p * 3:(p + 1) * 3])
            qc.compose(block, [src, snk], inplace=True)
            qc.barrier()
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        """
        Build the full QCNN ansatz with alternating convolution and pooling layers.
        """
        qc = QuantumCircuit(self.num_wires)
        # First conv
        qc.append(self._conv_layer(self.num_wires, "c1"), range(self.num_wires))
        # First pool
        qc.append(self._pool_layer(list(range(self.num_wires // 2)), list(range(self.num_wires // 2, self.num_wires)),
                                   "p1"), range(self.num_wires))
        # Second conv on the remaining qubits
        remaining = self.num_wires // 2
        qc.append(self._conv_layer(remaining, "c2"), range(remaining, self.num_wires))
        # Second pool
        qc.append(self._pool_layer(list(range(remaining // 2)), list(range(remaining // 2, remaining)),
                                   "p2"), range(remaining, self.num_wires))
        # Third conv
        qc.append(self._conv_layer(remaining // 2, "c3"), range(remaining // 2 + remaining, self.num_wires))
        # Third pool
        qc.append(self._pool_layer([0], [1], "p3"), range(remaining // 2 + remaining, self.num_wires))
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x
            Tensor of shape (batch_size, num_features).

        Returns
        -------
        torch.Tensor
            Regression output of shape (batch_size,).
        """
        # QNN returns expectation values of shape (batch_size, 1)
        q_output = self.qnn(x)
        # Pass through classical regression head
        out = self.head(q_output)
        return out.squeeze(-1)

def generate_superposition_data(num_wires: int, samples: int):
    """
    Generate data consisting of superposition states and corresponding labels.
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

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that returns quantum states and continuous targets.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32)
        }

__all__ = ["QCNNRegressionHybrid", "RegressionDataset", "generate_superposition_data"]
