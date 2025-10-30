import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumAttentionModule(tq.QuantumModule):
    """Quantum module that replaces the projection step of multi‑head attention."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(embed_dim)]
        )
        self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(embed_dim)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.parameters):
            gate(q_device, wires=wire)
        for wire in range(self.embed_dim - 1):
            tqf.cnot(q_device, wires=[wire, wire + 1])
        tqf.cnot(q_device, wires=[self.embed_dim - 1, 0])
        return self.measure(q_device)


class QuantumFeedForwardModule(tq.QuantumModule):
    """Quantum feed‑forward sub‑module producing an ffn‑dimensional vector."""
    def __init__(self, ffn_dim: int):
        super().__init__()
        self.ffn_dim = ffn_dim
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(ffn_dim)]
        )
        self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(ffn_dim)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.parameters):
            gate(q_device, wires=wire)
        for wire in range(self.ffn_dim - 1):
            tqf.cnot(q_device, wires=[wire, wire + 1])
        tqf.cnot(q_device, wires=[self.ffn_dim - 1, 0])
        return self.measure(q_device)


class QuantumAttentionWrapper:
    """Callable wrapper that adapts a quantum module for the hybrid attention."""
    def __init__(self, module: QuantumAttentionModule, device: torch.device = torch.device("cpu")) -> None:
        self.module = module
        self.device = device
        self.q_device = tq.QuantumDevice(n_wires=module.embed_dim, device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        out = self.module(x, self.q_device)
        return out.to(x.device)


class QuantumFeedForwardWrapper:
    """Callable wrapper that adapts a quantum module for the hybrid feed‑forward."""
    def __init__(self, module: QuantumFeedForwardModule, device: torch.device = torch.device("cpu")) -> None:
        self.module = module
        self.device = device
        self.q_device = tq.QuantumDevice(n_wires=module.ffn_dim, device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        out = self.module(x, self.q_device)
        return out.to(x.device)


__all__ = [
    "QuantumAttentionModule",
    "QuantumFeedForwardModule",
    "QuantumAttentionWrapper",
    "QuantumFeedForwardWrapper",
]
