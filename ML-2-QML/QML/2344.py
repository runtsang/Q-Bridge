import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def conv_circuit(params):
    """Two‑qubit convolutional unitary used in the QCNN."""
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

def conv_layer(num_qubits, param_prefix):
    """Builds a convolutional layer over pairs of qubits."""
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        layer = conv_circuit(params[param_index:param_index+3])
        qc.append(layer, [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def pool_circuit(params):
    """Two‑qubit pooling unitary."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources, sinks, param_prefix):
    """Pools information from source qubits into sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for source, sink in zip(sources, sinks):
        layer = pool_circuit(params[param_index:param_index+3])
        qc.append(layer, [source, sink])
        qc.barrier()
        param_index += 3
    return qc

class HybridQuanvolutionClassifier(nn.Module):
    """
    Quantum hybrid model that implements a QCNN with a feature map,
    convolutional and pooling layers, and a parameterised ansatz.
    The forward pass returns a log‑softmax over 10 classes.
    """
    def __init__(self, feature_dim: int = 10) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        # Feature map
        self.feature_map = ZFeatureMap(feature_dim)
        # Ansatz construction
        self.ansatz = QuantumCircuit(feature_dim)
        # Example: one convolution + one pooling
        self.ansatz.compose(conv_layer(feature_dim, "c1"), range(feature_dim), inplace=True)
        self.ansatz.compose(pool_layer(list(range(feature_dim//2)), list(range(feature_dim//2, feature_dim)), "p1"), range(feature_dim), inplace=True)
        # Combine feature map and ansatz
        self.circuit = QuantumCircuit(feature_dim)
        self.circuit.compose(self.feature_map, range(feature_dim), inplace=True)
        self.circuit.compose(self.ansatz, range(feature_dim), inplace=True)
        # Observables: one Z on each qubit to produce 10 outputs
        self.observables = [SparsePauliOp.from_list([("Z" + "I" * i + "I" * (feature_dim - i - 1), 1)]) for i in range(feature_dim)]
        # Estimator and QNN
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observables,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass. `x` should be a batch of feature vectors
        matching the input parameters of the feature map.
        """
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        # Compute expectation values
        output = self.qnn(x_np)
        # Convert to torch tensor and apply log‑softmax
        logits = torch.tensor(output, dtype=torch.float32)
        return torch.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionClassifier"]
