"""Quantum‑style kernel engine that leverages TorchQuantum and Qutip."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

# --- Quantum kernel utilities ------------------------------------------------- #

class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""

    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            tq.func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            tq.func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""

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

def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b``."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --- Quantum self‑attention --------------------------------------------------- #

class SelfAttention(tq.QuantumModule):
    """Basic quantum circuit representing a self‑attention style block."""

    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.qr = tq.QuantumRegister(self.n_qubits, "q")
        self.cr = tq.ClassicalRegister(self.n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> tq.QuantumCircuit:
        circuit = tq.QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
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
    ):
        # Dummy implementation to avoid circuit execution
        return {"00": shots // 2, "01": shots // 2}

# --- Quantum LSTM surrogate --------------------------------------------------- #

class QLSTM(tq.QuantumModule):
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
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
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

# --- Unified interface ---------------------------------------------------------- #

class QuantumKernelMethod:
    """
    Quantum‑enhanced kernel engine mirroring the classical surrogate.
    """

    def __init__(self, mode: str = "quantum", n_wires: int = 4, n_qubits: int = 4) -> None:
        self.mode = mode
        self.n_wires = n_wires
        self.n_qubits = n_qubits

        self.kernel = Kernel()
        self.attention_module = SelfAttention() if mode == "quantum" else None
        self.lstm_module = QLSTM(4, 4, n_qubits) if mode == "quantum" else None

    def _apply_attention(self, X: torch.Tensor) -> torch.Tensor:
        if self.attention_module is None:
            return X
        rot = np.random.randn(X.shape[1], 12)
        ent = np.random.randn(X.shape[1], 4)
        return self.attention_module.run(None, rot, ent, shots=1024)

    def _apply_lstm(self, seqs: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.lstm_module is None:
            return seqs
        hidden_states = []
        for seq in seqs:
            seq = seq.unsqueeze(1)  # batch dimension
            out, _ = self.lstm_module(seq)
            hidden_states.append(out.squeeze(1))
        return hidden_states

    def compute_kernel(
        self,
        a: list[torch.Tensor],
        b: list[torch.Tensor],
        data_type: str = "vector",
    ) -> np.ndarray:
        """
        Compute a Gram matrix between two collections of data.
        * vector – pair‑wise quantum kernel
        * sequence – average quantum kernel over hidden states
        """
        if data_type == "vector":
            a = self._apply_attention(a)
            b = self._apply_attention(b)
            return kernel_matrix(a, b)

        if data_type == "sequence":
            a_hidden = self._apply_lstm(a)
            b_hidden = self._apply_lstm(b)
            K = np.zeros((len(a_hidden), len(b_hidden)))
            for i, ha in enumerate(a_hidden):
                for j, hb in enumerate(b_hidden):
                    mat = kernel_matrix(ha, hb)
                    K[i, j] = mat.mean()
            return K

        raise ValueError(f"Unsupported data type: {data_type}")

__all__ = ["QuantumKernelMethod"]
