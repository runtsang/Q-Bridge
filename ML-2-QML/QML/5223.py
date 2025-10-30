import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

class QuantumHybridNAT(nn.Module):
    """Quantum variational network that mirrors the classical encoder.  It builds a QCNN‑style
    convolution‑pooling ansatz, concatenates a Z‑feature map that encodes the latent vector,
    and returns the expectation value of a Z observable on the first qubit.  The circuit
    is constructed using Qiskit primitives and wrapped in an EstimatorQNN for differentiable
    forward passes.
    """
    def __init__(self, num_qubits: int = 8):
        super().__init__()
        algorithm_globals.random_seed = 42
        self.num_qubits = num_qubits

        # Feature map that accepts the latent vector
        self.feature_map = ZFeatureMap(num_qubits)

        # Build QCNN‑style ansatz
        self.ansatz = self._build_ansatz(num_qubits)

        # Full circuit: feature map followed by ansatz
        full_circuit = QuantumCircuit(num_qubits)
        full_circuit.compose(self.feature_map, range(num_qubits), inplace=True)
        full_circuit.compose(self.ansatz, range(num_qubits), inplace=True)

        # Estimator for expectation values
        self.estimator = Estimator()

        # EstimatorQNN for differentiable quantum inference
        self.qnn = EstimatorQNN(
            circuit=full_circuit.decompose(),
            observables=SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)]),
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _build_ansatz(self, num_qubits: int) -> QuantumCircuit:
        """Constructs a QCNN‑style convolution‑pooling ansatz."""
        def conv_circuit(params):
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

        def pool_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        def conv_layer(num_qubits, param_prefix):
            qc = QuantumCircuit(num_qubits)
            qubits = list(range(num_qubits))
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits * 3)
            for q1, q2 in zip(qubits[0::2], qubits[1::2]):
                qc.append(conv_circuit(params[param_index: param_index + 3]), [q1, q2])
                qc.barrier()
                param_index += 3
            for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
                qc.append(conv_circuit(params[param_index: param_index + 3]), [q1, q2])
                qc.barrier()
                param_index += 3
            return qc

        def pool_layer(sources, sinks, param_prefix):
            num_qubits = len(sources) + len(sinks)
            qc = QuantumCircuit(num_qubits)
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            for source, sink in zip(sources, sinks):
                qc.append(pool_circuit(params[param_index: param_index + 3]), [source, sink])
                qc.barrier()
                param_index += 3
            return qc

        # Assemble full ansatz
        ansatz = QuantumCircuit(num_qubits)
        # First convolution layer
        ansatz.compose(conv_layer(num_qubits, "c1"), list(range(num_qubits)), inplace=True)
        # First pooling layer
        ansatz.compose(pool_layer(list(range(num_qubits // 2)), list(range(num_qubits // 2, num_qubits)), "p1"), list(range(num_qubits)), inplace=True)
        # Second convolution layer
        ansatz.compose(conv_layer(num_qubits // 2, "c2"), list(range(num_qubits // 2, num_qubits)), inplace=True)
        # Second pooling layer
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(num_qubits // 2, num_qubits)), inplace=True)
        # Third convolution layer
        ansatz.compose(conv_layer(num_qubits // 4, "c3"), list(range(num_qubits // 4, num_qubits // 2)), inplace=True)
        # Third pooling layer
        ansatz.compose(pool_layer([0], [1], "p3"), list(range(num_qubits // 4, num_qubits // 2)), inplace=True)

        return ansatz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the quantum circuit on a batch of latent vectors.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, num_qubits) representing the latent features
            produced by the classical encoder.

        Returns
        -------
        torch.Tensor
            Expectation values of the Z observable on the first qubit.
        """
        return self.qnn(x)
