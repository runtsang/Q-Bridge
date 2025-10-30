"""Hybrid kernel class that uses a quantum self‑attention block
and a TorchQuantum variational kernel.

The quantum implementation mirrors the classical API so that
the same training loop can be used.  The attention block is
implemented with Qiskit and returns a probability distribution
over qubits; this distribution is interpreted as attention
weights.  The weighted data is then fed into a variational
quantum kernel built with TorchQuantum.

Typical usage::

    from QuantumKernelMethod import QuantumKernelMethod
    model = QuantumKernelMethod(embed_dim=4, gamma=0.5)
    K = model(rotation_params, entangle_params, x, y)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

# --------------------------------------------------------------------------- #
#  Quantum attention helper (Qiskit)
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """Quantum circuit that emulates a self‑attention mechanism."""

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        # Rotation layer
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Entanglement layer
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """Execute the circuit and return probability distribution."""
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        # Convert counts to probabilities
        probs = np.zeros(self.n_qubits)
        for bitstring, cnt in counts.items():
            idx = int(bitstring, 2)
            probs[idx % self.n_qubits] += cnt
        probs /= shots
        return probs


# --------------------------------------------------------------------------- #
#  Quantum RBF kernel via TorchQuantum
# --------------------------------------------------------------------------- #
class KernalAnsatz(tq.QuantumModule):
    """Variational ansatz that encodes classical data."""

    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(
        self,
        q_device: tq.QuantumDevice,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = (
                x[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](
                q_device, wires=info["wires"], params=params
            )
        for info in reversed(self.func_list):
            params = (
                -y[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](
                q_device, wires=info["wires"], params=params
            )


class Kernel(tq.QuantumModule):
    """Quantum kernel module."""

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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


# --------------------------------------------------------------------------- #
#  Hybrid kernel class
# --------------------------------------------------------------------------- #
class QuantumKernelMethod:
    """
    Hybrid kernel that uses quantum self‑attention to weight samples
    before evaluating a TorchQuantum variational kernel.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input feature vectors.
    gamma : float, optional
        RBF scaling parameter (passed to the quantum kernel).
    """

    def __init__(self, embed_dim: int, gamma: float = 1.0) -> None:
        self.embed_dim = embed_dim
        self.gamma = gamma
        self.attention = QuantumSelfAttention(n_qubits=embed_dim)
        self.backend = Aer.get_backend("qasm_simulator")
        self.kernel = Kernel()

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the attention‑weighted quantum kernel matrix.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the rotation layer of the attention circuit.
        entangle_params : np.ndarray
            Parameters for the entanglement layer of the attention circuit.
        x, y : np.ndarray
            Input feature matrices of shape (n_samples, embed_dim).

        Returns
        -------
        np.ndarray
            Kernel matrix of shape (len(x), len(y)).
        """
        # 1. Quantum attention: obtain probability distribution for each sample
        x_att = np.array(
            [
                self.attention.run(
                    self.backend,
                    rotation_params,
                    entangle_params,
                )
                for _ in x
            ]
        )
        y_att = np.array(
            [
                self.attention.run(
                    self.backend,
                    rotation_params,
                    entangle_params,
                )
                for _ in y
            ]
        )

        # 2. Apply attention weights to the data
        x_weighted = x * x_att[:, None]
        y_weighted = y * y_att[:, None]

        # 3. Evaluate quantum kernel on weighted data
        k_mat = np.array(
            [
                [self.kernel(torch.as_tensor(xi), torch.as_tensor(yi)).item()]
                for xi in x_weighted
                for yi in y_weighted
            ]
        ).reshape(len(x_weighted), len(y_weighted))
        return k_mat


def kernel_matrix(
    a: Sequence[np.ndarray],
    b: Sequence[np.ndarray],
    embed_dim: int,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Convenience wrapper for the quantum hybrid kernel.

    Parameters
    ----------
    a, b : Sequence[np.ndarray]
        Feature vectors of shape (embed_dim,).
    embed_dim : int
        Dimensionality of the input vectors.
    gamma : float, optional
        RBF scaling parameter.

    Returns
    -------
    np.ndarray
        Gram matrix of shape (len(a), len(b)).
    """
    method = QuantumKernelMethod(embed_dim, gamma)
    rotation = np.random.randn(4 * embed_dim).reshape(4, embed_dim)
    entangle = np.random.randn(4 * embed_dim).reshape(4, embed_dim)
    return method(rotation, entangle, np.array(a), np.array(b))


__all__ = ["QuantumKernelMethod", "kernel_matrix"]
