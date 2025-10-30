"""Hybrid quantum convolution and attention module.

The quantum implementation mirrors the classical interface but uses a
quanvolution circuit followed by a parameter‑driven attention circuit
implemented with Qiskit.  The module can be executed on any backend
supporting the QASM simulator, and the number of shots can be tuned
to trade accuracy for speed.

The design follows a *combination* scaling paradigm: a quantum filter
produces a probability value that is then fed to a quantum attention
circuit.  The attention circuit is parameterised by rotation and
entangle parameters supplied by the caller, providing a direct
quantum‑classical interface.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit.random import random_circuit
from qiskit.providers import Backend


class HybridConvAttention:
    """
    Quantum hybrid of a quanvolution filter and a self‑attention block.

    Parameters
    ----------
    conv_kernel_size : int
        Size of the square quanvolution kernel.
    attention_dim : int
        Number of qubits used in the attention circuit.
    backend : qiskit.providers.Backend, optional
        Backend to execute the circuits.
    shots : int, optional
        Number of shots for each execution.
    conv_threshold : float, optional
        Threshold used to encode classical data into rotation angles.
    """

    def __init__(
        self,
        conv_kernel_size: int = 2,
        attention_dim: int = 4,
        backend: Backend | None = None,
        shots: int = 1024,
        conv_threshold: float = 127.0,
    ) -> None:
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.conv_kernel_size = conv_kernel_size
        self.attention_dim = attention_dim
        self.conv_circuit = self._build_quanv(conv_kernel_size, conv_threshold)
        self.n_qubits_conv = conv_kernel_size ** 2
        self.n_qubits_attn = attention_dim
        self.conv_threshold = conv_threshold

    def _build_quanv(self, kernel_size: int, threshold: float) -> QuantumCircuit:
        n_qubits = kernel_size ** 2
        qc = QuantumCircuit(n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += random_circuit(n_qubits, 2)
        qc.measure_all()
        self.theta = theta
        return qc

    def _build_attention(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits_attn, "q")
        cr = ClassicalRegister(self.n_qubits_attn, "c")
        qc = QuantumCircuit(qr, cr)
        for i in range(self.n_qubits_attn):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits_attn - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure_all()
        return qc

    def run(
        self,
        data: np.ndarray,
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
    ) -> float:
        """
        Execute the hybrid quantum circuit on the given data.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape ``(kernel_size, kernel_size)``.
        rotation_params : np.ndarray, optional
            Parameters for the rotation part of the attention circuit.
        entangle_params : np.ndarray, optional
            Parameters for the entanglement part of the attention circuit.

        Returns
        -------
        float
            Expected probability of measuring |1> across all qubits
            after the attention circuit, scaled by the quanvolution output.
        """
        # Quanvolution step
        flat = np.reshape(data, (1, self.n_qubits_conv))
        binds = []
        for val in flat[0]:
            angle = np.pi if val > self.conv_threshold else 0.0
            binds.append({self.theta[i]: angle for i in range(self.n_qubits_conv)})

        job = execute(
            self.conv_circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=binds,
        )
        result = job.result().get_counts(self.conv_circuit)
        conv_counts = sum(
            sum(int(bit) for bit in key) * count
            for key, count in result.items()
        )
        conv_prob = conv_counts / (self.shots * self.n_qubits_conv)

        # Attention step
        if rotation_params is None:
            rotation_params = np.random.randn(self.n_qubits_attn * 3)
        if entangle_params is None:
            entangle_params = np.random.randn(self.n_qubits_attn - 1)

        attn_circuit = self._build_attention(rotation_params, entangle_params)

        job = execute(
            attn_circuit,
            backend=self.backend,
            shots=self.shots,
        )
        result = job.result().get_counts(attn_circuit)
        att_counts = sum(
            sum(int(bit) for bit in key) * count
            for key, count in result.items()
        )
        att_prob = att_counts / (self.shots * self.n_qubits_attn)

        # Combine the two probabilities multiplicatively to emulate
        # a feature‑wise weighting mechanism.
        return conv_prob * att_prob


__all__ = ["HybridConvAttention"]
