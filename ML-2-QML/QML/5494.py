from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Quantum infrastructure
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.primitives import StatevectorEstimator as Estimator
from qiskit.circuit.library import ZFeatureMap

# ------------------------------------------------------------------
# Quantum Self‑Attention (Qiskit implementation)
# ------------------------------------------------------------------
class QuantumSelfAttention:
    """
    Builds a small parameterised circuit that mimics a self‑attention
    operation.  The returned counts are later interpreted as a
    feature vector.
    """
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits)
        self.cr = ClassicalRegister(n_qubits)
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)


# ------------------------------------------------------------------
# Quantum QCNN (full‑stack circuit)
# ------------------------------------------------------------------
def QuantumQCNN() -> EstimatorQNN:
    """
    Returns a QNN instance that implements the QCNN from the
    reference.  The circuit is built from the original
    conv/pool primitives and wrapped in an EstimatorQNN.
    """
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

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

    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    def pool_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for src, snk in zip(sources, sinks):
            qc.compose(pool_circuit(params[param_index : param_index + 3]), [src, snk])
            qc.barrier()
            param_index += 3
        return qc

    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8)
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


# ------------------------------------------------------------------
# Quantum Fully‑Connected Layer
# ------------------------------------------------------------------
class QuantumFullyConnected:
    """
    Simple parameterised circuit that measures a single qubit
    expectation value.  The ``run`` method matches the classical
    FullyConnectedLayer interface.
    """
    def __init__(self):
        self.circuit = qiskit.QuantumCircuit(1)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 100

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys()), dtype=float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])


# ------------------------------------------------------------------
# Quantum‑enhanced Sequence Tagger
# ------------------------------------------------------------------
class SharedClassName(nn.Module):
    """
    Quantum‑enhanced sequence‑tagger that parallels the classical
    SharedClassName.  Every classical sub‑module is replaced by its
    quantum counterpart while the LSTM backbone remains classical
    to keep the model trainable on non‑quantum hardware.
    """
    def __init__(
        self,
        vocab_size: int,
        tagset_size: int,
        embed_dim: int = 8,
        hidden_dim: int = 32,
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.self_attention = QuantumSelfAttention(n_qubits)
        self.qcnn = QuantumQCNN()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = QuantumFullyConnected()
        self.tag_layer = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        # 1. Embedding
        embeds = self.embedding(sentence)  # (seq_len, embed_dim)

        # 2. Quantum self‑attention
        rotation_params = np.random.randn(embeds.shape[1] * 3)
        entangle_params = np.random.randn(embeds.shape[1] * 3)
        counts = self.self_attention.run(rotation_params, entangle_params)
        # Convert counts to a feature vector (simple expectation)
        attn_vec = np.zeros(embeds.shape[1])
        for state, cnt in counts.items():
            attn_vec[int(state, 2)] = cnt
        attn_tensor = torch.as_tensor(attn_vec, device=embeds.device).unsqueeze(0)

        # 3. QCNN feature extraction (token‑wise)
        qcnn_feats = []
        for token in attn_tensor:
            pred = self.qcnn.predict(token.cpu().numpy().reshape(1, -1))
            qcnn_feats.append(torch.tensor(pred, device=token.device).squeeze(0))
        qcnn_tensor = torch.stack(qcnn_feats, dim=0).unsqueeze(0)  # (1, seq_len, 1)

        # 4. LSTM
        lstm_out, _ = self.lstm(qcnn_tensor)

        # 5. Fully‑connected quantum head
        fc_out = self.fc.run(lstm_out.squeeze(0).cpu().numpy())
        fc_tensor = torch.as_tensor(fc_out, device=lstm_out.device)

        # 6. Tag projection
        logits = self.tag_layer(fc_tensor)
        return F.log_softmax(logits, dim=1)


__all__ = ["SharedClassName"]
