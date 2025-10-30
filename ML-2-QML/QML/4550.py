"""Hybrid quantum network combining quantum convolution, self‑attention, and QCNN layers."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# Import subcomponents
from Conv import Conv as QuanvCircuit
from SelfAttention import SelfAttention
from QCNN import QCNN as QCNN_Qiskit

class HybridConvNet:
    """Quantum‑inspired hybrid network mirroring the classical HybridConvNet.

    It processes an image through a variational convolution circuit, a quantum
    self‑attention block, and a QCNN‑style ansatz, returning a scalar probability.
    """

    def __init__(self, kernel_size: int = 2, conv_shots: int = 100, conv_threshold: float = 127) -> None:
        self.backend = Aer.get_backend("qasm_simulator")

        # 1. Convolution layer
        self.conv = QuanvCircuit(kernel_size, self.backend, shots=conv_shots, threshold=conv_threshold)

        # 2. Self‑attention
        self.attention = SelfAttention()

        # 3. QCNN ansatz
        self.qcnn = QCNN_Qiskit()
        self.qcnn_estimator = self.qcnn  # EstimatorQNN instance

    def _patch_features(self, image: np.ndarray) -> np.ndarray:
        """Return a 1‑D array of feature values obtained by applying the quantum
        convolution circuit to every 2×2 patch of the image."""
        h, w = image.shape
        patch_size = int(self.conv.n_qubits ** 0.5)
        features = []
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = image[i : i + patch_size, j : j + patch_size]
                if patch.shape!= (patch_size, patch_size):
                    patch = np.pad(patch, ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1])))
                feat = self.conv.run(patch)
                features.append(feat)
        return np.array(features)

    def _attention_features(self, features: np.ndarray) -> np.ndarray:
        """Map the patch features to a 4‑dimensional vector using the quantum
        self‑attention circuit."""
        rotation_params = np.random.randn(12)
        entangle_params = np.random.randn(3)
        result = self.attention.run(
            self.backend,
            rotation_params,
            entangle_params,
            shots=1024,
        )
        probs = np.zeros(4)
        total = 0
        for bitstring, count in result.items():
            total += count
            for idx, bit in enumerate(reversed(bitstring)):
                probs[idx] += int(bit) * count
        probs = probs / total
        return probs

    def run(self, image: np.ndarray) -> float:
        """Execute the hybrid quantum network and return a probability in [0, 1]."""
        # 1. Convolution features
        conv_feats = self._patch_features(image)

        # 2. Self‑attention mapping
        attn_feats = self._attention_features(conv_feats)

        # 3. Build QCNN input vector of length 8
        conv_avg = conv_feats.mean()
        qcnn_input = np.concatenate([np.full(4, conv_avg), attn_feats])

        # 4. Run QCNN
        input_dict = {name: val for name, val in zip(self.qcnn.input_params, qcnn_input)}
        probs = self.qcnn_estimator.predict([input_dict])[0]
        return float(probs)

__all__ = ["HybridConvNet"]
