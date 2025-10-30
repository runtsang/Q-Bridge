"""Hybrid kernel–attention model – quantum implementation."""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import AerSimulator

# --------------------------------------------------------------------------- #
# Quantum kernel ansatz
# --------------------------------------------------------------------------- #
class KernalAnsatz(tq.QuantumModule):
    """Programmable list of gates encoding classical data."""

    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = (
                x[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = (
                -y[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


# --------------------------------------------------------------------------- #
# Hybrid model
# --------------------------------------------------------------------------- #
class KernelAttentionModel(tq.QuantumModule):
    """Quantum kernel combined with a Qiskit self‑attention block."""

    def __init__(self, embed_dim: int = 4, gamma: float = 1.0, n_qubits: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.gamma = gamma
        self.n_qubits = n_qubits

        # Quantum device for the kernel
        self.q_device = tq.QuantumDevice(n_wires=self.n_qubits)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        # Qiskit backend for attention
        self.backend = AerSimulator()
        self._setup_attention_circuit()

    # --------------------------------------------------------------------------- #
    # Quantum kernel
    # --------------------------------------------------------------------------- #
    def compute_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the absolute overlap from the variational ansatz."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    # --------------------------------------------------------------------------- #
    # Quantum self‑attention
    # --------------------------------------------------------------------------- #
    def _setup_attention_circuit(self):
        """Prepare a reusable Qiskit circuit template."""
        self.qr = QuantumRegister(self.n_qubits, "q")
        self.cr = ClassicalRegister(self.n_qubits, "c")
        self.base_circuit = QuantumCircuit(self.qr, self.cr)

    def _build_circuit(self, rotation_params, entangle_params):
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)
            circuit.rz(entangle_params[i], i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def compute_attention(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """Run the attention circuit and return a probability vector."""
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=shots)
        counts = job.result().get_counts(circuit)
        # Convert counts to a probability distribution over the n_qubits‑bitstring
        probs = np.zeros(2**self.n_qubits)
        for bitstring, cnt in counts.items():
            idx = int(bitstring[::-1], 2)  # Qiskit returns little‑endian
            probs[idx] = cnt / shots
        return probs

    # --------------------------------------------------------------------------- #
    # Matrix helpers
    # --------------------------------------------------------------------------- #
    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Gram matrix of the quantum kernel."""
        return np.array(
            [
                [self.compute_kernel(torch.tensor(x, dtype=torch.float32),
                                     torch.tensor(y, dtype=torch.float32))
                 for y in b]
                for x in a
            ]
        )

    def combined_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """Combine the quantum kernel matrix with attention probabilities."""
        K = self.kernel_matrix(a, b)
        attn = self.compute_attention(rotation_params, entangle_params, shots=shots)
        # Use the first len(a) probabilities as row weights
        weights = attn[: len(a)]
        return K * weights[:, None]


__all__ = ["KernelAttentionModel"]
