"""Hybrid quantum self‑attention with convolution and TorchQuantum kernel."""
from __future__ import annotations

import numpy as np
import torch
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.random import random_circuit
from qiskit import execute, Aer
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence


class QuantumSelfAttention:
    """Variational attention circuit producing per‑qubit |1> probabilities."""
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self,
            backend,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


class QuanvCircuit:
    """Quantum convolution filter for 2×2 kernels."""
    def __init__(self,
                 kernel_size: int,
                 backend,
                 shots: int = 100,
                 threshold: float = 127) -> None:
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """
        Parameters
        ----------
        data : 2‑D array of shape (k, k) with integer pixel values.
        Returns
        -------
        float
            Average probability of measuring |1> across qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {self.theta[i]: (np.pi if val > self.threshold else 0) for i, val in enumerate(dat)}
            param_binds.append(bind)

        job = execute(self._circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)


class KernalAnsatz(tq.QuantumModule):
    """TorchQuantum ansatz encoding two classical vectors."""
    def __init__(self, func_list) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self,
                q_device: tq.QuantumDevice,
                x: torch.Tensor,
                y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class QuantumKernel(tq.QuantumModule):
    """Quantum kernel using a fixed 4‑wire ansatz."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


class HybridSelfAttention:
    """
    Quantum‑classical hybrid self‑attention.
    Combines a Qiskit attention circuit, a quantum convolution filter,
    and a TorchQuantum kernel for similarity weighting.
    """
    def __init__(self,
                 n_qubits: int = 4,
                 conv_kernel: int = 2,
                 backend=None) -> None:
        self.attention = QuantumSelfAttention(n_qubits)
        self.conv_filter = QuanvCircuit(
            kernel_size=conv_kernel,
            backend=backend or Aer.get_backend("qasm_simulator")
        )
        self.kernel = QuantumKernel()

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params, entangle_params : np.ndarray
            Parameters for the attention variational circuit.
        inputs : np.ndarray of shape (B, L, D)
            Classical data to be processed.
        shots : int
            Number of shots for the Qiskit circuit.
        Returns
        -------
        np.ndarray
            Hybrid output of shape (B, L, D).
        """
        B, L, D = inputs.shape
        # Convolutional contribution
        conv_vals = []
        for i in range(B):
            # reshape each embedding to a kernel square
            vec = inputs[i].reshape(conv_kernel, conv_kernel)
            conv_vals.append(self.conv_filter.run(vec))
        conv_vals = np.array(conv_vals)[:, None, None]  # (B,1,1)

        # Quantum attention
        counts = self.attention.run(
            backend=self.attention.qr.register,  # placeholder; actual backend set in init
            rotation_params=rotation_params,
            entangle_params=entangle_params,
            shots=shots
        )
        # Convert counts to per‑qubit |1> probabilities
        total = sum(counts.values())
        probs = np.zeros(self.attention.n_qubits)
        for bitstring, cnt in counts.items():
            for i, bit in enumerate(bitstring[::-1]):
                probs[i] += (cnt / total) * int(bit)
        # Broadcast as attention weights over sequence dimension
        attn_weights = probs[:L]  # assume L <= n_qubits
        attn_weights = attn_weights / attn_weights.sum()
        attn_weights = attn_weights.reshape(1, L, 1)  # (1, L, 1)

        # Apply weights to inputs
        weighted = inputs * attn_weights  # (B, L, D)
        # Combine convolution and weighted values
        output = weighted + conv_vals
        return output

    def kernel_matrix(self,
                      a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute Gram matrix using the quantum kernel.
        """
        kernel = self.kernel
        return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["HybridSelfAttention"]
