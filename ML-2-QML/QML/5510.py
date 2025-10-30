import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate states of the form cos(θ)|0…0⟩ + e^{iϕ} sin(θ)|1…1⟩
    and a target derived from θ and ϕ.
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
    return states, labels

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that returns a dictionary with a complex state tensor and a float target.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumRegression(tq.QuantumModule):
    """
    Variational quantum regression model that combines a QCNN‑style ansatz
    with a linear head.  The ansatz is built from convolution and pooling
    layers inspired by the QCNN reference.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.qnn = self._build_qnn()
        self.head = nn.Linear(num_wires, 1)

    # ------------------------------------------------------------------
    # QCNN‑style ansatz construction
    # ------------------------------------------------------------------
    @staticmethod
    def conv_circuit(params: ParameterVector) -> QuantumCircuit:
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
        return target

    @staticmethod
    def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.compose(QuantumRegression.conv_circuit(params[param_index:param_index+3]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.compose(QuantumRegression.conv_circuit(params[param_index:param_index+3]), [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    @staticmethod
    def pool_circuit(params: ParameterVector) -> QuantumCircuit:
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    @staticmethod
    def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for src, snk in zip(sources, sinks):
            qc.compose(QuantumRegression.pool_circuit(params[param_index:param_index+3]), [src, snk])
            qc.barrier()
            param_index += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        ansatz = QuantumCircuit(self.n_wires, name="Ansatz")
        # First Convolutional Layer
        ansatz.compose(self.conv_layer(self.n_wires, "c1"), list(range(self.n_wires)), inplace=True)
        # First Pooling Layer
        ansatz.compose(self.pool_layer(list(range(self.n_wires//2)), list(range(self.n_wires//2, self.n_wires)), "p1"),
                       list(range(self.n_wires)), inplace=True)
        # Second Convolutional Layer
        ansatz.compose(self.conv_layer(self.n_wires//2, "c2"), list(range(self.n_wires//2, self.n_wires)), inplace=True)
        # Second Pooling Layer
        ansatz.compose(self.pool_layer([0,1], [2,3], "p2"), list(range(self.n_wires//2, self.n_wires)), inplace=True)
        return ansatz

    # ------------------------------------------------------------------
    # EstimatorQNN wrapper
    # ------------------------------------------------------------------
    def _build_qnn(self) -> EstimatorQNN:
        feature_map = ZFeatureMap(self.n_wires)
        ansatz = self._build_ansatz()
        circuit = QuantumCircuit(self.n_wires)
        circuit.compose(feature_map, range(self.n_wires), inplace=True)
        circuit.compose(ansatz, range(self.n_wires), inplace=True)
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_wires - 1), 1)])
        estimator = Estimator()
        return EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        state_batch: shape (batch, 2**n_wires) complex tensor
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        # Expectation values from the estimator
        features = self.qnn(state_batch)
        return self.head(features).squeeze(-1)

__all__ = ["QuantumRegression", "RegressionDataset", "generate_superposition_data"]
