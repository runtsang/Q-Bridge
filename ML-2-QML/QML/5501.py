import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.functional import func_name_dict


def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap of two measurement vectors, normalised."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: torch.Tensor,
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> torch.Tensor:
    """
    Build a weighted adjacency matrix from pairwise state fidelities.
    Edges with fidelity >= threshold get weight 1.0, secondary values get
    secondary_weight if provided.
    """
    n = states.shape[0]
    adj = torch.zeros((n, n), dtype=torch.float32, device=states.device)
    for i in range(n):
        for j in range(i + 1, n):
            fid = state_fidelity(states[i], states[j])
            if fid >= threshold:
                adj[i, j] = adj[j, i] = 1.0
            elif secondary is not None and fid >= secondary:
                adj[i, j] = adj[j, i] = secondary_weight
    return adj


class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data into a programmable list of quantum gates."""

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


class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""

    def __init__(self):
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


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head for quantum expectations."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class HybridQuantumNAT(tq.QuantumModule):
    """
    Quantum hybrid model mirroring the classical architecture:
      * TorchQuantum encoder and parameterised QLayer.
      * Measurement‑based state vectors.
      * Fidelity‑based adjacency weighting.
      * Quantum kernel similarity.
      * Differentiable sigmoid head producing a two‑class probability.
    """

    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        self.kernel = Kernel()
        self.shift = shift
        self.hybrid = HybridFunction.apply

    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
            )
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.norm(out)

        # Build adjacency from measurement fidelities
        adj = fidelity_adjacency(out, threshold=0.9)
        weighted_out = torch.mm(adj, out)

        # Quantum kernel similarity
        kernel_out = self.kernel.forward(out, weighted_out).unsqueeze(1)

        # Concatenate and aggregate
        combined = torch.cat([out, weighted_out, kernel_out], dim=1)
        logits = combined.sum(dim=1, keepdim=True)
        probs = self.hybrid(logits, self.shift)
        return torch.cat([probs, 1 - probs], dim=-1)


__all__ = ["HybridQuantumNAT"]
