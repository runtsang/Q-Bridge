import numpy as np
from qiskit import Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.primitives import Estimator as EstimatorQ
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_circuit(params: ParameterVector) -> qiskit.circuit.QuantumCircuit:
    """Two‑qubit convolution block used in QCNN ansatz."""
    qc = qiskit.circuit.QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc


def _pool_circuit(params: ParameterVector) -> qiskit.circuit.QuantumCircuit:
    """Two‑qubit pooling block used in QCNN ansatz."""
    qc = qiskit.circuit.QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _conv_layer(num_qubits: int, prefix: str) -> qiskit.circuit.QuantumCircuit:
    """Compose convolution layers across qubit pairs."""
    qc = qiskit.circuit.QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        sub = _conv_circuit(params[param_index:param_index+3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        sub = _conv_circuit(params[param_index:param_index+3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    return qc


def _pool_layer(sources: list[int], sinks: list[int], prefix: str) -> qiskit.circuit.QuantumCircuit:
    """Compose pooling layers across specified source‑sink pairs."""
    num_qubits = len(sources) + len(sinks)
    qc = qiskit.circuit.QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    for s, t in zip(sources, sinks):
        sub = _pool_circuit(params[param_index:param_index+3])
        qc.append(sub, [s, t])
        qc.barrier()
        param_index += 3
    return qc


class HybridCNNQML(nn.Module):
    """
    Quantum‑only implementation of the QCNN architecture.
    The network accepts an 8‑dimensional feature vector, encodes it with a ZFeatureMap,
    then applies a stack of convolution and pooling layers, and finally measures
    the expectation of the Z observable on the first qubit.
    """
    def __init__(self,
                 n_qubits: int = 8,
                 shots: int = 1024,
                 shift: float = np.pi / 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.shift = shift
        self.backend = Aer.get_backend("aer_simulator")
        self.estimator = EstimatorQ()
        self.qnn = self._build_qnn()

    def _build_qnn(self) -> EstimatorQNN:
        feature_map = ZFeatureMap(self.n_qubits)
        ansatz = self._build_ansatz()
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1)])
        return EstimatorQNN(
            circuit=ansatz.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=self.estimator,
        )

    def _build_ansatz(self) -> qiskit.circuit.QuantumCircuit:
        # Top‑level ansatz: feature map + three conv‑pool stages
        circuit = qiskit.circuit.QuantumCircuit(self.n_qubits)
        circuit.compose(ZFeatureMap(self.n_qubits), range(self.n_qubits), inplace=True)

        # Stage 1
        circuit.compose(_conv_layer(self.n_qubits, "c1"), range(self.n_qubits), inplace=True)
        circuit.compose(_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(self.n_qubits), inplace=True)

        # Stage 2
        circuit.compose(_conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        circuit.compose(_pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

        # Stage 3
        circuit.compose(_conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        circuit.compose(_pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

        return circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass using the EstimatorQNN.  Inputs should be a tensor of shape
        (batch, n_qubits) containing angles for the feature map.
        Returns a tensor of shape (batch, 2) with class probabilities.
        """
        # Convert to numpy for the estimator
        if isinstance(inputs, torch.Tensor):
            inputs_np = inputs.detach().cpu().numpy()
        else:
            inputs_np = np.asarray(inputs)
        expectation = self.qnn(inputs_np).reshape(-1, 1)
        probs = torch.tensor(expectation, dtype=torch.float32)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridCNNQML"]
