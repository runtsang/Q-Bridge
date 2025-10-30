"""
Quantum counterpart of SelfAttentionHybrid.
Encodes image patches into qubits, applies a quantum kernel,
then evaluates a self‑attention‑like correlation via measurement statistics.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter

# ---- 1. Parameters for fraud‑detection style layer (quantum analog) ----
class FraudLayerParameters:
    """Quantum‑friendly parameters mirroring the classical fraud‑detection layer."""
    def __init__(
        self,
        theta: float,
        phi: float,
        r: float,
        phi_r: float,
        k: float,
    ):
        self.theta = theta
        self.phi = phi
        self.r = r
        self.phi_r = phi_r
        self.k = k


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


# ---- 2. Quantum kernel acting on 2‑pixel patches ----
class QuantumKernel:
    """Applies a small parameterized circuit to two‑qubit patches."""
    def __init__(self, n_qubits: int = 2):
        self.n_qubits = n_qubits
        self.params = {f"theta_{i}": Parameter(f"θ_{i}") for i in range(n_qubits)}
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qreg = QuantumRegister(self.n_qubits, "q")
        circuit = QuantumCircuit(qreg, name="kernel")
        for i in range(self.n_qubits):
            circuit.ry(self.params[f"theta_{i}"], qreg[i])
            circuit.rz(self.params[f"theta_{i}"], qreg[i])
        # entangle
        circuit.cx(qreg[0], qreg[1])
        circuit.measure_all()
        return circuit

    def bind_params(self, data: np.ndarray) -> QuantumCircuit:
        """Bind classical data to rotation angles (data ∈ [0, 2π])."""
        bound = {self.params[f"theta_{i}"]: float(data[i]) for i in range(self.n_qubits)}
        return self.circuit.bind_parameters(bound)


# ---- 3. Quantum self‑attention block ----
class QuantumSelfAttention:
    """
    Quantum circuit that emulates a self‑attention style operation.
    Encodes each patch, entangles adjacent patches, and measures correlation.
    """
    def __init__(self, n_patches: int, n_qubits_per_patch: int = 2):
        self.n_patches = n_patches
        self.n_qubits_per_patch = n_qubits_per_patch
        self.total_qubits = n_patches * n_qubits_per_patch
        self.qreg = QuantumRegister(self.total_qubits, "q")
        self.creg = ClassicalRegister(self.total_qubits, "c")
        self.circuit = QuantumCircuit(self.qreg, self.creg, name="self_attention")
        self.kernel = QuantumKernel(n_qubits_per_patch)
        self._build()

    def _build(self) -> None:
        # Encode each patch with its own kernel
        for p in range(self.n_patches):
            start = p * self.n_qubits_per_patch
            sub_circ = self.kernel.circuit
            for i in range(self.n_qubits_per_patch):
                self.circuit.append(
                    sub_circ.data[i][0],  # gate
                    [self.qreg[start + i], *[self.qreg[start + i]]],
                )
        # Entangle adjacent patches (CRX style)
        for p in range(self.n_patches - 1):
            idx = p * self.n_qubits_per_patch
            next_idx = (p + 1) * self.n_qubits_per_patch
            self.circuit.crx(np.pi / 4, self.qreg[idx + 0], self.qreg[next_idx + 0])
        # Measure all qubits
        self.circuit.measure(self.qreg, self.creg)

    def run(
        self,
        backend,
        patch_data: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """
        Execute the circuit for a batch of patch data.
        :param patch_data: shape (N_patches, n_qubits_per_patch) with values in [0, 2π]
        :return: measurement counts
        """
        job = execute(self.circuit, backend, shots=shots, parameter_binds=[{k: float(v) for k, v in zip(self.kernel.params.values(), patch_data.flatten())}])
        return job.result().get_counts(self.circuit)


# ---- 4. Hybrid SelfAttentionHybrid quantum version ----
class SelfAttentionHybrid:
    """
    Quantum implementation mirroring SelfAttentionHybrid.
    Encodes image patches, applies a fraud‑detection style kernel,
    then runs a self‑attention‑like measurement.
    """
    def __init__(self, n_patches: int = 196, n_qubits_per_patch: int = 2):
        self.attention_block = QuantumSelfAttention(n_patches, n_qubits_per_patch)
        self.backend = Aer.get_backend("qasm_simulator")

    def run(
        self,
        image: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """
        :param image: 2‑D array of shape (H, W) with values in [0, 2π]
        :return: measurement dictionary
        """
        # 1. Extract 2×2 patches
        patches = []
        H, W = image.shape
        for r in range(0, H, 2):
            for c in range(0, W, 2):
                patch = image[r : r + 2, c : c + 2].flatten()
                patches.append(patch)
        patch_data = np.array(patches)  # shape (N, n_qubits_per_patch)
        return self.attention_block.run(self.backend, patch_data, shots)
__all__ = ["SelfAttentionHybrid"]
