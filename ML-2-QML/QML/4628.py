import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import Aer
from qiskit.circuit.random import random_circuit
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# ---------- Quantum filter (quanvolution) ----------
class QuanvCircuit:
    """Quantum filter that processes a 2×2 patch."""
    def __init__(self, kernel_size, backend, shots, threshold):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data):
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(self._circuit,
                            self.backend,
                            shots=self.shots,
                            parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

def Conv() -> QuanvCircuit:
    backend = qiskit.Aer.get_backend("qasm_simulator")
    filter_size = 2
    return QuanvCircuit(filter_size, backend, shots=100, threshold=127)

# ---------- QCNN ansatz ----------
def conv_circuit(params):
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

def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

def QCNN() -> EstimatorQNN:
    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8, name="Ansatz")

    # First Convolutional Layer
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
    # First Pooling Layer
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
    # Second Convolutional Layer
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
    # Second Pooling Layer
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
    # Third Convolutional Layer
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
    # Third Pooling Layer
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
    estimator = Estimator()
    return EstimatorQNN(
        circuit=ansatz.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )

# ---------- Hybrid quantum‑classical classifier ----------
class QuantumHybridQCNNConvNet(nn.Module):
    """
    Quantum‑enhanced hybrid network that mirrors the classical
    HybridQCNNConvNet. It uses a 2×2 quantum filter and a QCNN ansatz
    for feature extraction, followed by a classical sigmoid head.
    """
    def __init__(self):
        super().__init__()
        self.conv_filter = Conv()
        self.qcnn = QCNN()
        self.backend = Aer.get_backend("aer_simulator")
        self.fc = nn.Linear(1, 1)

    def _extract_patches(self, imgs: torch.Tensor) -> torch.Tensor:
        B, C, H, W = imgs.shape
        patches = imgs.unfold(2, 2, 1).unfold(3, 2, 1)
        patches = patches.contiguous().view(B, C, -1, 2, 2)
        patches = patches.mean(1)  # collapse channel dim
        return patches  # (B, num_patches, 2, 2)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        patches = self._extract_patches(imgs)  # (B, N, 2, 2)
        B, N, _, _ = patches.shape
        probs = []
        for b in range(B):
            patch_probs = []
            for n in range(N):
                patch = patches[b, n].cpu().numpy()
                patch_probs.append(self.conv_filter.run(patch))
            probs.append(patch_probs)
        probs = torch.tensor(probs, dtype=torch.float)  # (B, N)
        # Duplicate each probability to match 8‑dim input of QCNN
        inputs = probs.unsqueeze(-1).repeat(1, 1, 8)  # (B, N, 8)
        logits = self.qcnn(inputs)  # (B, N, 1)
        out = logits.mean(1)  # (B, 1)
        prob = torch.sigmoid(self.fc(out))
        return torch.cat([prob, 1 - prob], dim=-1)


__all__ = ["QuantumHybridQCNNConvNet"]
