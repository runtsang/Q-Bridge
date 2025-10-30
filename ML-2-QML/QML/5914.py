"""Quantum hybrid attention and quanvolution module.

This module builds on the SelfAttention and Quanvolution seeds.  It
encodes each 2×2 pixel patch into four qubits, applies a parameterized
rotation circuit, entangles neighboring patches with CRX gates, and then
measures all qubits.  The resulting measurement pattern can be used as
a quantum kernel for downstream classical processing.

The `SelfAttentionQuanvolutionHybridQuantum` class mirrors the interface
of the classical hybrid but runs a variational circuit on a Qiskit
simulator.
"""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

class SelfAttentionQuanvolutionHybridQuantum:
    def __init__(self, n_patches: int = 196, n_qubits_per_patch: int = 4):
        self.n_patches = n_patches
        self.n_qubits_per_patch = n_qubits_per_patch
        self.total_qubits = n_patches * n_qubits_per_patch
        self.qr = QuantumRegister(self.total_qubits, "q")
        self.cr = ClassicalRegister(self.total_qubits, "c")

    def _build_circuit(self, image: np.ndarray, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = image[r:r+2, c:c+2].flatten()
                patches.append(patch)
        for i, patch in enumerate(patches):
            base = i * self.n_qubits_per_patch
            for q in range(self.n_qubits_per_patch):
                idx = base + q
                r_angle = rotation_params[3 * idx]
                ry_angle = rotation_params[3 * idx + 1]
                rz_angle = rotation_params[3 * idx + 2]
                qc.rx(r_angle, idx)
                qc.ry(ry_angle, idx)
                qc.rz(rz_angle, idx)
        for i in range(self.n_patches - 1):
            src = i * self.n_qubits_per_patch + self.n_qubits_per_patch - 1
            dst = (i + 1) * self.n_qubits_per_patch
            qc.crx(entangle_params[i], src, dst)
        qc.measure(self.qr, self.cr)
        return qc

    def run(self, image: np.ndarray, backend=Aer.get_backend("qasm_simulator"),
            rotation_params: np.ndarray = None, entangle_params: np.ndarray = None,
            shots: int = 1024):
        if rotation_params is None:
            rotation_params = np.random.uniform(0, 2 * np.pi, size=self.total_qubits * 3)
        if entangle_params is None:
            entangle_params = np.random.uniform(0, np.pi, size=self.n_patches - 1)
        qc = self._build_circuit(image, rotation_params, entangle_params)
        job = execute(qc, backend, shots=shots)
        return job.result().get_counts(qc)

__all__ = ["SelfAttentionQuanvolutionHybridQuantum"]
