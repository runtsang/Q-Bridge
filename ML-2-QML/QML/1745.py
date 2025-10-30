import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATEnhanced(tq.QuantumModule):
    """Enhanced hybrid quantum model for multi‑output regression.

    The module augments the original Quantum‑NAT circuit with a
    feature‑wise attention encoder and a parameterised entangling ansatz.
    A measurement of all qubits in the Pauli‑Z basis yields four
    expectation values that are normalised with a BatchNorm1d layer.
    The class exposes a ``predict`` helper and a ``set_device`` method
    for quick inference and backend switching.
    """

    class FeatureAttention(tq.QuantumModule):
        """Learned linear scaling of pooled features before encoding."""
        def __init__(self, in_features: int, n_wires: int):
            super().__init__()
            self.scale = nn.Linear(in_features, n_wires, bias=False)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice, pooled: torch.Tensor):
            """Apply learned scaling and encode with a simple 4‑qubit encoder."""
            scaled = self.scale(pooled)
            # Simple Ry/Rz rotations per qubit
            for i in range(4):
                tqf.ry(qdev, wires=i, params=scaled[:, i],
                       static=self.static_mode, parent_graph=self.graph)
                tqf.rz(qdev, wires=i, params=scaled[:, i],
                       static=self.static_mode, parent_graph=self.graph)
            # Entangle with a ladder of CNOTs
            for i in range(3):
                tqf.cnot(qdev, wires=[i, i+1],
                         static=self.static_mode, parent_graph=self.graph)

    class VariationalAnsatz(tq.QuantumModule):
        """Parameterised entangling circuit."""
        def __init__(self, n_wires: int, depth: int = 3):
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            # Parameterised single‑qubit rotations
            self.rxs = nn.ParameterList([nn.Parameter(torch.randn(n_wires))
                                         for _ in range(depth)])
            self.rys = nn.ParameterList([nn.Parameter(torch.randn(n_wires))
                                         for _ in range(depth)])
            self.rzs = nn.ParameterList([nn.Parameter(torch.randn(n_wires))
                                         for _ in range(depth)])

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            for d in range(self.depth):
                tqf.rx(qdev, wires=list(range(self.n_wires)),
                       params=self.rxs[d], static=self.static_mode,
                       parent_graph=self.graph)
                tqf.ry(qdev, wires=list(range(self.n_wires)),
                       params=self.rys[d], static=self.static_mode,
                       parent_graph=self.graph)
                tqf.rz(qdev, wires=list(range(self.n_wires)),
                       params=self.rzs[d], static=self.static_mode,
                       parent_graph=self.graph)
                # Entangling layer
                for i in range(self.n_wires - 1):
                    tqf.cz(qdev, wires=[i, i+1], static=self.static_mode,
                           parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Encoder maps 16‑dim pooled features to qubit angles
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.attention = self.FeatureAttention(in_features=16, n_wires=self.n_wires)
        self.q_layer = self.VariationalAnsatz(n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def set_device(self, device: torch.device) -> "QuantumNATEnhanced":
        """Move quantum device and parameters to the specified device."""
        return self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz,
                                device=x.device, record_op=True)
        # Extract 16‑dim pooled features
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.attention(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience inference wrapper for the quantum model."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

__all__ = ["QuantumNATEnhanced"]
