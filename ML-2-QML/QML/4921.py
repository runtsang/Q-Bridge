"""Hybrid quantum layer that mirrors HybridQuantumLayer but implements each block with Qiskit."""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from typing import Iterable

__all__ = ["HybridQuantumLayer"]


class _FullyConnectedQuantum:
    """Parameterized Ry on a single qubit, expectation value of Z."""
    def __init__(self, n_qubits: int = 1, shots: int = 200):
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.circuit = QuantumCircuit(n_qubits)
        self.theta = Parameter("θ")
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        param_binds = [{self.theta: t} for t in thetas]
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = {k: v / self.shots for k, v in counts.items()}
        # Map |1⟩ to +1, |0⟩ to -1 for Z expectation
        expectation = sum((1 if b == '1' else -1) * p for b, p in probs.items())
        return np.array([expectation])


class _ConvQuantum:
    """4‑qubit circuit acting on a 2×2 patch, returns average |1⟩ probability."""
    def __init__(self, n_qubits: int = 4, shots: int = 200, threshold: float = 0.5):
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self.circuit = QuantumCircuit(n_qubits)
        self.theta = [Parameter(f"θ{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        # Random two‑qubit entangling layer (simple example)
        self.circuit.cx(0, 1)
        self.circuit.cx(2, 3)
        self.circuit.measure_all()

    def run(self, patch: np.ndarray) -> np.ndarray:
        """`patch` is a (4,) vector with values in [0,1]."""
        binds = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(patch)}
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[binds])
        result = job.result()
        counts = result.get_counts(self.circuit)
        # Probability of measuring |1⟩ across all qubits
        total_ones = sum(sum(int(b) for b in key) * cnt for key, cnt in counts.items())
        prob = total_ones / (self.shots * len(self.theta))
        return np.array([prob])


class _QuanvolutionQuantum:
    """Applies the 2×2 quantum filter over every patch of a 28×28 image."""
    def __init__(self, patch_size: int = 2, stride: int = 2, shots: int = 200):
        self.patch_size = patch_size
        self.stride = stride
        self.filter = _ConvQuantum(n_qubits=patch_size**2, shots=shots)

    def run(self, image: np.ndarray) -> np.ndarray:
        """`image` shape (28,28)."""
        patches = []
        for r in range(0, 28, self.stride):
            for c in range(0, 28, self.stride):
                patch = image[r:r+self.patch_size, c:c+self.patch_size].flatten()
                patches.append(self.filter.run(patch))
        return np.concatenate(patches)  # shape (14*14,)


class _QNatQuantum:
    """A small 4‑qubit random circuit mimicking the Quantum‑NAT head."""
    def __init__(self, shots: int = 200):
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.circuit = QuantumCircuit(4)
        # Random entangling layer
        self.circuit.h(range(4))
        self.circuit.cx(0, 1)
        self.circuit.cx(2, 3)
        self.circuit.measure_all()

    def run(self, features: np.ndarray) -> np.ndarray:
        """`features` shape (n,4) – flattened patches from quanvolution."""
        # For simplicity, treat each feature vector as a single shot
        probs = []
        for feat in features:
            binds = {f"q{i}": int(b) for i, b in enumerate(feat)}
            job = execute(self.circuit, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(self.circuit)
            # Map measurement to a 4‑dim vector (use counts of each basis state)
            vec = np.zeros(4)
            for key, cnt in counts.items():
                idx = int(key, 2)
                vec[idx] = cnt
            probs.append(vec / self.shots)
        return np.concatenate(probs)  # shape (n*4,)


class HybridQuantumLayer:
    """
    Quantum implementation of the hybrid layer:
      * Fully‑connected block (single‑qubit Ry circuit)
      * 2×2 quantum convolution (4‑qubit circuit)
      * Quanvolution (tile‑wise application of the convolution)
      * Quantum‑NAT head (4‑qubit random circuit)
    The `run` method accepts either:
      * `data` as a list of angles for the FC block (mode='fc')
      * a 28×28 image (mode='img')
    """
    def __init__(self, shots: int = 200):
        self.fc = _FullyConnectedQuantum(shots=shots)
        self.quanv = _QuanvolutionQuantum(shots=shots)
        self.qnat = _QNatQuantum(shots=shots)

    def run(self, data, mode: str = "fc") -> np.ndarray:
        if mode == "fc":
            return self.fc.run(data)
        elif mode == "img":
            # Assume data shape (28,28) or (1,28,28)
            if data.ndim == 3:
                data = data.squeeze(0)
            # Apply quanvolution
            patch_features = self.quanv.run(data)  # shape (196,)
            # Reshape into (14*14, 4) for QNat
            patches = patch_features.reshape(-1, 4)
            nat_out = self.qnat.run(patches)
            return nat_out
        else:
            raise ValueError(f"Unsupported mode: {mode}")
