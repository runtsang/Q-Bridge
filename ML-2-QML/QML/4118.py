"""Quantum‑enhanced Self‑Attention combining a variational circuit, a quantum‑style convolution, and a classical LSTM gate."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvCircuit:
    """Quantum convolution filter (quanvolution) used as a feature extractor."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: int):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Run the quantum convolution on a 2×2 data block."""
        data = data.reshape(1, self.n_qubits)
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(self._circuit,
                             self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)


class QuantumLSTMGate(nn.Module):
    """Classical LSTM gate used to modulate quantum attention scores."""
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        gate = torch.sigmoid(self.linear(out))
        return gate


class SelfAttention(nn.Module):
    """
    Quantum‑enhanced self‑attention that uses a variational circuit for attention scoring,
    a quantum convolution for feature extraction, and a classical LSTM gate for dynamic
    weighting of the attention scores.
    """
    def __init__(self,
                 n_qubits: int = 4,
                 conv_kernel: int = 2,
                 lstm_hidden: int = 32):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.conv = QuanvCircuit(kernel_size=conv_kernel,
                                 backend=self.backend,
                                 shots=100,
                                 threshold=127)
        self.lstm_gate = QuantumLSTMGate(hidden_dim=lstm_hidden)

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure_all()
        return circuit

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            data: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (n_qubits*3,) – rotation angles for each qubit.
        entangle_params : np.ndarray
            Shape (n_qubits-1,) – angles for controlled‑RX gates.
        data : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Attended representation of shape (batch, n_qubits).
        """
        batch, seq_len, _ = data.shape
        results = []

        for b in range(batch):
            seq_out = []
            for t in range(seq_len):
                # Quantum attention via measurement probabilities
                circuit = self._build_circuit(rotation_params, entangle_params)
                job = qiskit.execute(circuit, self.backend, shots=1024)
                counts = job.result().get_counts(circuit)
                probs = np.zeros(self.n_qubits)
                for bitstring, cnt in counts.items():
                    ones = sum(int(bit) for bit in bitstring)
                    probs += ones * cnt
                probs = probs / (1024 * self.n_qubits)

                # Convolutional feature
                conv_feat = self.conv.run(data[b, t, :].reshape(2, 2))
                conv_tensor = torch.tensor(conv_feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                gate = self.lstm_gate(conv_tensor).item()

                gated = probs * gate
                seq_out.append(gated)

            seq_out = np.mean(seq_out, axis=0)  # aggregate over sequence
            results.append(seq_out)

        return np.stack(results, axis=0)

__all__ = ["SelfAttention"]
