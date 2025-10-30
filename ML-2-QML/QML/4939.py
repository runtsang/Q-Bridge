import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class ClassicalSelfAttention:
    """A lightweight self‑attention module that mimics the quantum‑style interface."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        query = np.dot(inputs, rotation_params.reshape(self.embed_dim, -1))
        key   = np.dot(inputs, entangle_params.reshape(self.embed_dim, -1))
        scores = np.exp(query @ key.T / np.sqrt(self.embed_dim))
        scores /= scores.sum(axis=-1, keepdims=True)
        return scores @ inputs

class QCNNGen144:
    """Quantum hybrid QCNN that embeds convolution, pooling and attention circuits, and outputs a sampler‑style probability."""
    def __init__(self, num_qubits: int = 8, embed_dim: int = 4):
        self.num_qubits = num_qubits
        self.embed_dim = embed_dim

        # Feature map (classical embedding into the quantum state)
        self.feature_map = ZFeatureMap(num_qubits)

        # Ansatz: convolution + pooling + attention
        self.ansatz = QuantumCircuit(num_qubits, name="QCNNAnsatz")
        self.ansatz.compose(self.conv_layer(num_qubits, "c1"), inplace=True)
        self.ansatz.compose(self.pool_layer([0,1,2,3],[4,5,6,7],"p1"), inplace=True)
        self.ansatz.compose(self.conv_layer(num_qubits//2, "c2"), inplace=True)
        self.ansatz.compose(self.pool_layer([0,1],[2,3],"p2"), inplace=True)
        self.ansatz.compose(self.conv_layer(num_qubits//4, "c3"), inplace=True)
        self.ansatz.compose(self.pool_layer([0],[1],"p3"), inplace=True)
        self.ansatz.compose(self.attention_circuit(embed_dim, "att"), inplace=True)

        # Observable for a simple classification output
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits-1), 1)])

        # Sampler that will provide a probability distribution over bitstrings
        self.sampler = StatevectorSampler()
        self.qnn_sampler = SamplerQNN(
            circuit=self.ansatz,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            sampler=self.sampler,
        )

    def conv_layer(self, n_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        params = ParameterVector(f"{prefix}_", length=n_qubits*3)
        for i in range(n_qubits):
            qc.rx(params[3*i], i)
            qc.ry(params[3*i+1], i)
            qc.rz(params[3*i+2], i)
        for i in range(0, n_qubits-1, 2):
            qc.cx(i, i+1)
        return qc

    def pool_layer(self, sources, sinks, prefix: str) -> QuantumCircuit:
        n = len(sources) + len(sinks)
        qc = QuantumCircuit(n)
        params = ParameterVector(f"{prefix}_", length=n//2*3)
        for idx, (s, t) in enumerate(zip(sources, sinks)):
            for j in range(3):
                qc.rz(params[3*idx+j], s)
            qc.cx(s, t)
        return qc

    def attention_circuit(self, embed_dim: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        rot_params = ParameterVector(f"{prefix}_rot", length=3*embed_dim)
        ent_params = ParameterVector(f"{prefix}_ent", length=embed_dim-1)
        for i in range(embed_dim):
            qc.rx(rot_params[3*i], i)
            qc.ry(rot_params[3*i+1], i)
            qc.rz(rot_params[3*i+2], i)
        for i in range(embed_dim-1):
            qc.crx(ent_params[i], i, i+1)
        return qc

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Predict returns a dict mapping bitstrings to probabilities
        probs = self.qnn_sampler.predict(inputs)
        # Interpret the first qubit as the class label
        class_prob = sum(p for b,p in probs.items() if b[0] == '1')
        return np.array([class_prob], dtype=np.float32)

__all__ = ["QCNNGen144"]
