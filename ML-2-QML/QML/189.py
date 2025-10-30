import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit_machine_learning.neural_networks import EstimatorQNN
import torch
from torch import nn

class QCNNModel(nn.Module):
    """
    Hybrid quantum‑classical QCNN implementation.

    The class builds a QNN with a feature‑map and a stack of
    convolutional + pooling layers implemented as parameterised two‑qubit
    blocks.  The resulting EstimatorQNN is wrapped in a PyTorch
    nn.Module so it can be trained with standard optimisers.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the feature map (default 8).
    backend : str
        Backend name for the Estimator (default 'aer_simulator').
    """

    def __init__(self, num_qubits: int = 8, backend: str = "aer_simulator"):
        super().__init__()
        self.num_qubits = num_qubits
        self.backend = backend

        # Feature map
        self.feature_map = ZFeatureMap(num_qubits)

        # Ansatz construction
        self.ansatz = self._build_ansatz(num_qubits)

        # Observables
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

        # Estimator and QNN
        provider = AerSimulator()
        quantum_instance = QuantumInstance(provider)
        self.estimator = Estimator(quantum_instance=quantum_instance)
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _build_ansatz(self, num_qubits: int) -> QuantumCircuit:
        """Builds a stack of convolution + pooling layers as in the QCNN paper."""
        def conv_block(qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(prefix, 3 * len(qubits) // 2)
            idx = 0
            for i in range(0, len(qubits) - 1, 2):
                sub = self._conv_subcircuit(params[idx:idx+3], qubits[i], qubits[i+1])
                qc.append(sub, [qubits[i], qubits[i+1]])
                qc.barrier()
                idx += 3
            return qc

        def pool_block(qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(prefix, 3 * len(qubits) // 2)
            idx = 0
            for i in range(0, len(qubits) - 1, 2):
                sub = self._pool_subcircuit(params[idx:idx+3], qubits[i], qubits[i+1])
                qc.append(sub, [qubits[i], qubits[i+1]])
                qc.barrier()
                idx += 3
            return qc

        qc = QuantumCircuit(num_qubits)
        # Layer 1
        qc.append(conv_block(range(num_qubits), "c1"), range(num_qubits))
        # Pool 1
        qc.append(pool_block(range(num_qubits), "p1"), range(num_qubits))
        # Layer 2
        qc.append(conv_block(range(num_qubits // 2, num_qubits), "c2"), range(num_qubits // 2, num_qubits))
        # Pool 2
        qc.append(pool_block(range(num_qubits // 2, num_qubits), "p2"), range(num_qubits // 2, num_qubits))
        # Layer 3
        qc.append(conv_block(range(num_qubits // 4, num_qubits // 2), "c3"), range(num_qubits // 4, num_qubits // 2))
        # Pool 3
        qc.append(pool_block(range(num_qubits // 4, num_qubits // 2), "p3"), range(num_qubits // 4, num_qubits // 2))
        return qc

    def _conv_subcircuit(self, params, q1, q2):
        """Two‑qubit convolution sub‑circuit used in the QCNN ansatz."""
        sub = QuantumCircuit(2)
        sub.rz(-np.pi/2, 1)
        sub.cx(1, 0)
        sub.rz(params[0], 0)
        sub.ry(params[1], 1)
        sub.cx(0, 1)
        sub.ry(params[2], 1)
        sub.cx(1, 0)
        sub.rz(np.pi/2, 0)
        return sub

    def _pool_subcircuit(self, params, q1, q2):
        """Two‑qubit pooling sub‑circuit used in the QCNN ansatz."""
        sub = QuantumCircuit(2)
        sub.rz(-np.pi/2, 1)
        sub.cx(1, 0)
        sub.rz(params[0], 0)
        sub.ry(params[1], 1)
        sub.cx(0, 1)
        sub.ry(params[2], 1)
        return sub

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the QNN wrapped in a PyTorch module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, num_qubits).
        Returns
        -------
        torch.Tensor
            Predicted probabilities of shape (batch, 1).
        """
        # The EstimatorQNN expects a torch tensor with shape (batch, num_inputs)
        out = self.qnn(x)
        return torch.sigmoid(out)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Convenience wrapper for NumPy inputs."""
        return self.qnn.predict(x)
