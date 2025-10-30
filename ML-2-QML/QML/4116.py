"""Hybrid quantum module that mirrors the classical SelfAttentionHybrid.

The quantum implementation contains:
* A self‑attention circuit built with Qiskit parameterised rotations
  and controlled‑X entanglement.
* A quanvolution filter that performs a random circuit on a kernel‑size
  grid of qubits, with classical data encoded into rotation angles.
* A simple fidelity‑based kernel that evaluates the overlap between
  two data‑encoded states.

The public API keeps the same `run` method as the classical
reference, returning a dictionary of results for each sub‑module.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.random import random_circuit
import torch
from typing import Sequence

class QuantumSelfAttention:
    """Parameterised Qiskit circuit that implements a self‑attention block."""
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build(self, rot_params: np.ndarray, ent_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        # rotation layer
        for i in range(self.n_qubits):
            qc.rx(rot_params[3 * i], i)
            qc.ry(rot_params[3 * i + 1], i)
            qc.rz(rot_params[3 * i + 2], i)
        # entanglement layer
        for i in range(self.n_qubits - 1):
            qc.crx(ent_params[i], i, i + 1)
        qc.measure_all()
        return qc

    def run(self, backend, rot_params: np.ndarray, ent_params: np.ndarray,
            shots: int = 1024) -> dict:
        qc = self._build(rot_params, ent_params)
        job = execute(qc, backend, shots=shots)
        return job.result().get_counts(qc)

class QuanvCircuit:
    """Quantum convolution filter that maps a 2‑D patch to a scalar."""
    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100,
                 threshold: float = 127.0):
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Encode the 2‑D data patch into rotation angles and evaluate."""
        flat = data.reshape(1, self.n_qubits)
        binds = []
        for row in flat:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(row)}
            binds.append(bind)
        job = execute(self._circuit, self.backend, shots=self.shots,
                      parameter_binds=binds)
        result = job.result().get_counts(self._circuit)
        # average number of |1> outcomes
        total = 0
        for key, val in result.items():
            total += val * sum(int(b) for b in key)
        return total / (self.shots * self.n_qubits)

class QuantumKernel:
    """Simple fidelity‑based kernel using a fixed entangling circuit."""
    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.base_circuit = self._build_base()

    def _build_base(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.h(i)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        return qc

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        qc = self._build_base()
        for i, val in enumerate(data):
            qc.ry(val, i)
        return qc

    def kernel_value(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return the overlap between |x> and |y>."""
        qc_x = self.encode(x)
        qc_y = self.encode(y)
        # swap test circuit
        swap_qc = QuantumCircuit(self.n_qubits + 1)
        swap_qc.h(0)
        for i in range(self.n_qubits):
            swap_qc.cx(i, i + 1)
            swap_qc.cx(0, i + 1)
            swap_qc.cx(i, i + 1)
        swap_qc += qc_x
        swap_qc += qc_y
        swap_qc.h(0)
        swap_qc.measure(0, 0)
        job = execute(swap_qc, self.backend, shots=self.shots)
        counts = job.result().get_counts(swap_qc)
        return counts.get('0', 0) / self.shots

    def kernel_matrix(self, a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
        return np.array([[self.kernel_value(x, y) for y in b] for x in a])

class SelfAttentionHybrid:
    """Quantum counterpart of the classical SelfAttentionHybrid."""
    def __init__(self,
                 n_qubits: int = 4,
                 kernel_size: int = 2,
                 threshold: float = 127.0,
                 shots: int = 1024):
        self.attention = QuantumSelfAttention(n_qubits)
        self.quanv = QuanvCircuit(kernel_size, shots=shots, threshold=threshold)
        self.kernel = QuantumKernel(n_qubits, shots=shots)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> dict:
        """
        Execute all three quantum sub‑modules and return their outputs.

        Returns
        -------
        dict
            {
                'attention': counts dict,
                'quanv': scalar probability,
                'kernel': 2‑D Gram matrix
            }
        """
        attn_out = self.attention.run(Aer.get_backend("qasm_simulator"),
                                      rotation_params, entangle_params)
        quanv_out = self.quanv.run(inputs[0])  # assume first sample is a patch
        kernel_out = self.kernel.kernel_matrix([inputs[0]], [inputs[0]])
        return {
            'attention': attn_out,
            'quanv': quanv_out,
            'kernel': kernel_out
        }

def SelfAttention() -> SelfAttentionHybrid:
    """Factory that mirrors the classical API."""
    return SelfAttentionHybrid()
