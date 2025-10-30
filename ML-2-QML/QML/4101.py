"""Hybrid quantum model combining quantum kernel, quantum LSTM, and quantum autoencoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Sequence

# --------------------------------------------------------------------------- #
# Quantum kernel components (TorchQuantum)
# --------------------------------------------------------------------------- #
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumKernelAnsatz(tq.QuantumModule):
    """Programmable list of quantum gates for data encoding."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumKernelAnsatz(
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

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b``."""
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Quantum autoencoder (Qiskit)
# --------------------------------------------------------------------------- #
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
import qiskit as qk

class QuantumAutoencoder:
    """Variational autoencoder based on a RealAmplitudes ansatz and a swap‑test."""
    def __init__(self, input_dim: int, latent_dim: int = 3, trash_dim: int = 2) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.sampler = Sampler()
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.latent_dim + 2 * self.trash_dim + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        ansatz = RealAmplitudes(self.latent_dim + self.trash_dim, reps=5)
        circuit.compose(ansatz, range(0, self.latent_dim + self.trash_dim), inplace=True)
        circuit.barrier()
        aux = self.latent_dim + 2 * self.trash_dim
        circuit.h(aux)
        for i in range(self.trash_dim):
            circuit.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])
        return circuit

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of feature vectors.
        Note: For simplicity we flatten the input and feed it into the QNN.
        """
        flat = inputs.view(-1, self.input_dim)
        return self.qnn(flat)

# --------------------------------------------------------------------------- #
# Quantum LSTM component (TorchQuantum)
# --------------------------------------------------------------------------- #
class QuantumQLSTM(nn.Module):
    """LSTM cell where gates are realised by small quantum circuits."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tq.f.cnot(qdev, wires=[wire, 0])
                else:
                    tq.f.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# --------------------------------------------------------------------------- #
# Hybrid quantum model
# --------------------------------------------------------------------------- #
class HybridKernelAutoLSTM(nn.Module):
    """Hybrid pipeline that encodes data with a quantum autoencoder,
    compares them with a quantum kernel, and tags sequences with a quantum LSTM."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int = 3,
        n_qubits: int = 4,
        vocab_size: int = 1000,
        tagset_size: int = 10,
    ) -> None:
        super().__init__()
        self.kernel = QuantumKernel()
        self.autoencoder = QuantumAutoencoder(input_dim, latent_dim=latent_dim)
        self.lstm = QuantumQLSTM(input_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of feature vectors using the quantum autoencoder."""
        return self.autoencoder.encode(inputs)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Compute the kernel Gram matrix between two feature tensors."""
        a_flat = a.view(-1, a.size(-1))
        b_flat = b.view(-1, b.size(-1))
        return kernel_matrix(a_flat, b_flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, seq_len, input_dim)``.
        Returns
        -------
        torch.Tensor
            Log‑softmaxed tag logits of shape ``(batch, seq_len, tagset_size)``.
        """
        # Encode each token in the sequence
        encoded = self.encode(x.view(-1, x.size(-1))).view(x.size(0), x.size(1), -1)
        # Run through quantum LSTM
        lstm_out, _ = self.lstm(encoded)
        logits = self.hidden2tag(lstm_out)
        return torch.log_softmax(logits, dim=-1)

__all__ = [
    "HybridKernelAutoLSTM",
    "kernel_matrix",
    "QuantumAutoencoder",
    "QuantumKernel",
    "QuantumKernelAnsatz",
    "QuantumQLSTM",
    "QuantumQLSTM.QLayer",
]
