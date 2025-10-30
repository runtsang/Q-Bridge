import torch
import torch.nn as nn
import torch.nn.functional as F
from QLSTM import QLSTM as ClassicalQLSTM

class FraudGateLayer(nn.Module):
    """Linear + tanh gate with trainable scale and shift, inspired by fraud‑detection layers."""
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.activation = nn.Tanh()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x)) * self.scale + self.shift

class Kernel(nn.Module):
    """Radial basis function kernel used for similarity scoring."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class HybridQLSTM(ClassicalQLSTM):
    """
    Hybrid LSTM that augments the classical QLSTM with fraud‑detection style gates
    and a kernel‑based similarity multiplier applied to the forget gate.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        reference_vectors: list[torch.Tensor] | None = None,
        gamma: float = 1.0,
    ) -> None:
        super().__init__(input_dim, hidden_dim, n_qubits)
        # replace linear gates with fraud‑style gates
        self.forget_gate = FraudGateLayer(input_dim + hidden_dim)
        self.input_gate  = FraudGateLayer(input_dim + hidden_dim)
        self.update_gate = FraudGateLayer(input_dim + hidden_dim)
        self.output_gate = FraudGateLayer(input_dim + hidden_dim)
        self.kernel = Kernel(gamma)
        self.reference_vectors = reference_vectors or []

    def _kernel_score(self, x: torch.Tensor) -> torch.Tensor:
        if not self.reference_vectors:
            return torch.ones_like(x[:, 0])
        scores = [self.kernel(x, r) for r in self.reference_vectors]
        return torch.mean(torch.stack(scores, dim=0), dim=0)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update_gate(combined))
            o = torch.sigmoid(self.output_gate(combined))
            # modulate forget gate by similarity
            sim = self._kernel_score(combined)
            f = f * sim
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

__all__ = ["HybridQLSTM"]
