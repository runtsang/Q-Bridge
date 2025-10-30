import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import networkx as nx
import numpy as np

class QuantumRBFKernel(tq.QuantumModule):
    """
    Simple quantum kernel that encodes a data vector via Ry gates
    and measures the expectation value of Pauli‑Z on the first qubit.
    """
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor, p: torch.Tensor) -> None:
        self.encoder(qdev, x)
        for wire, val in enumerate(p):
            tq.RX(has_params=True, trainable=False)(qdev, wires=[wire], params=val)
        self.measure(qdev)

class QLSTM(nn.Module):
    """Quantum LSTM cell with gates implemented by small variational circuits."""
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

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] = None):
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
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class HybridFCL(nn.Module):
    """
    Quantum‑enhanced fully connected layer that merges kernel trick,
    graph regularisation and optional QLSTM for sequences.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_prototypes: int = 64,
                 n_qubits: int = 4,
                 seq: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_prototypes = n_prototypes
        self.n_qubits = n_qubits
        self.seq = seq

        # Prototype parameters for kernel expansion
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, n_qubits))

        # Quantum kernel module
        self.kernel = QuantumRBFKernel(n_qubits)

        # Linear mapping from kernel feature to hidden dimension
        self.linear = nn.Linear(n_prototypes, hidden_dim)

        # Optional QLSTM for sequences
        if seq:
            self.lstm = QLSTM(input_dim=hidden_dim,
                              hidden_dim=hidden_dim,
                              n_qubits=n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.seq:
            b, t, d = x.shape
            x_flat = x.reshape(b * t, d)
            feats = self._quantum_kernel(x_flat, self.prototypes)
            feats = feats.reshape(b, t, -1)
            feats = self.linear(feats)
            out, _ = self.lstm(feats)
            return out
        else:
            feats = self._quantum_kernel(x, self.prototypes)
            feats = self.linear(feats)
            return feats

    def _quantum_kernel(self, x: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        n = prototypes.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=batch, device=x.device)
        kernel_vals = torch.empty(batch, n, device=x.device)
        for i in range(n):
            qdev.reset_states(batch)
            self.kernel.encoder(qdev, x[:, :self.n_qubits])
            for wire, val in enumerate(prototypes[i]):
                tq.RX(has_params=True, trainable=False)(qdev, wires=[wire], params=val)
            out = self.kernel.measure(qdev)
            kernel_vals[:, i] = out[:, 0].real
        return kernel_vals

    def compute_adjacency(self,
                          states: torch.Tensor,
                          threshold: float = 0.8,
                          secondary: float | None = None,
                          secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        n = states.shape[0]
        graph.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                fid = torch.abs(torch.dot(states[i], states[j])) ** 2
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph

__all__ = ["HybridFCL"]
