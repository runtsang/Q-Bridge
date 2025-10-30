from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

class QuantumSelfAttention:
    """Variational circuit computing attention weights via expectation values."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = qiskit.Aer.get_backend("statevector_simulator")
        self.estimator = Estimator()

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> np.ndarray:
        circuit = self._build_circuit(rotation_params, entangle_params)
        obs = SparsePauliOp.from_list([("Z" * self.n_qubits, 1)])
        qnn = EstimatorQNN(
            circuit=circuit,
            observables=obs,
            input_params=[],
            weight_params=[],
            estimator=self.estimator,
        )
        return qnn.predict([])

class QuantumQuanvolutionFilter:
    """Quantum kernel applied to 2×2 image patches."""
    def __init__(self, n_wires: int = 4):
        self.n_wires = n_wires
        self.backend = qiskit.Aer.get_backend("statevector_simulator")
        self.estimator = Estimator()

    def _patch_circuit(self, data: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_wires)
        for i, val in enumerate(data):
            qc.initialize([np.cos(val), np.sin(val)], i)
        for i in range(self.n_wires - 1):
            qc.cx(i, i + 1)
        return qc

    def run(self, image: np.ndarray) -> np.ndarray:
        # image shape: (B, 1, H, W)
        B, _, H, W = image.shape
        patches = []
        for r in range(0, H, 2):
            for c in range(0, W, 2):
                data = image[:, 0, r:r+2, c:c+2].reshape(B, -1)
                patch_features = []
                for b in range(B):
                    qc = self._patch_circuit(data[b])
                    result = self.estimator.run(qc, self.backend)
                    exp_val = result.get_expectation_value("Z" * self.n_wires)
                    patch_features.append(exp_val)
                patches.append(patch_features)
        return np.array(patches).reshape(B, -1)

class QuantumQCNN:
    """Hybrid quantum convolution‑pooling stack using parameterized gates."""
    def __init__(self, num_qubits: int = 8):
        self.num_qubits = num_qubits
        self.backend = qiskit.Aer.get_backend("statevector_simulator")
        self.estimator = Estimator()

    def _conv_layer(self, params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        for i in range(0, self.num_qubits, 2):
            qc.cx(i, i+1)
            qc.rz(params[3 * (i//2)], i)
            qc.ry(params[3 * (i//2) + 1], i+1)
            qc.cx(i+1, i)
        return qc

    def _pool_layer(self, params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        for i in range(0, self.num_qubits-1, 2):
            qc.cx(i, i+1)
            qc.rz(params[3 * (i//2)], i)
            qc.ry(params[3 * (i//2) + 1], i+1)
        return qc

    def run(self, inputs: np.ndarray, conv_params: np.ndarray, pool_params: np.ndarray) -> np.ndarray:
        # inputs shape: (B, num_qubits)
        B = inputs.shape[0]
        outputs = []
        for b in range(B):
            qc = QuantumCircuit(self.num_qubits)
            for i, val in enumerate(inputs[b]):
                qc.initialize([np.cos(val), np.sin(val)], i)
            qc.append(self._conv_layer(conv_params), list(range(self.num_qubits)))
            qc.append(self._pool_layer(pool_params), list(range(self.num_qubits)))
            result = self.estimator.run(qc, self.backend)
            exp_vals = [result.get_expectation_value("Z" * self.num_qubits)]
            outputs.append(exp_vals)
        return np.array(outputs).reshape(B, -1)

class HybridAttentionNetwork:
    """
    Quantum‑backed hybrid attention network.

    * QuantumSelfAttention produces attention logits via a variational ansatz.
    * QuantumQuanvolutionFilter extracts quantum‑kernel features from image patches.
    * QuantumQCNN implements a convolution‑pooling hierarchy in circuit form.
    * The class exposes a unified `run` method that returns a dictionary of all
      quantum sub‑modules, ready for downstream classical processing.
    """
    def __init__(self, embed_dim: int = 4, num_qubits: int = 8, use_quantum: bool = True):
        self.embed_dim = embed_dim
        self.num_qubits = num_qubits
        self.use_quantum = use_quantum
        if use_quantum:
            self.attention = QuantumSelfAttention(n_qubits=embed_dim)
            self.quanvolution = QuantumQuanvolutionFilter()
            self.qcnn = QuantumQCNN(num_qubits=num_qubits)
        else:
            self.attention = None
            self.quanvolution = None
            self.qcnn = None

    def run(self,
            rotation_params: np.ndarray | None = None,
            entangle_params: np.ndarray | None = None,
            image: np.ndarray | None = None,
            conv_params: np.ndarray | None = None,
            pool_params: np.ndarray | None = None) -> dict:
        if not self.use_quantum:
            return {"info": "Quantum components disabled"}
        out = {}
        if self.attention:
            out["attention"] = self.attention.run(rotation_params, entangle_params)
        if self.quanvolution and image is not None:
            out["quanv"] = self.quanvolution.run(image)
        if self.qcnn and conv_params is not None and pool_params is not None:
            # For simplicity, use a random input vector of appropriate shape
            dummy_input = np.random.rand(1, self.num_qubits)
            out["qcnn"] = self.qcnn.run(dummy_input, conv_params, pool_params)
        return out

__all__ = ["HybridAttentionNetwork"]
