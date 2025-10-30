import torch
import torch.nn as nn
import torch.nn.functional as F

class FCLayer(nn.Module):
    """Classical approximation of a quantum fully‑connected layer."""
    def __init__(self, in_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        # thetas shape (batch, dim)
        out = torch.tanh(self.linear(thetas))  # shape (batch, 1)
        return out.mean(dim=1, keepdim=True)  # scalar per batch

class SamplerNet(nn.Module):
    """Small MLP that mimics a quantum sampler – outputs a probability vector."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

class QLSTMGen108(nn.Module):
    """Hybrid LSTM that uses classical circuits inspired by quantum layers."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # learnable scaling and shift per gate – emulates photonic clipping
        self.forget_scale = nn.Parameter(torch.ones(hidden_dim))
        self.forget_shift = nn.Parameter(torch.zeros(hidden_dim))
        self.input_scale = nn.Parameter(torch.ones(hidden_dim))
        self.input_shift = nn.Parameter(torch.zeros(hidden_dim))
        self.update_scale = nn.Parameter(torch.ones(hidden_dim))
        self.update_shift = nn.Parameter(torch.zeros(hidden_dim))
        self.output_scale = nn.Parameter(torch.ones(hidden_dim))
        self.output_shift = nn.Parameter(torch.zeros(hidden_dim))

        self.sampler_net = SamplerNet(hidden_dim)
        self.fcl = FCLayer()

    def _clip(self, x: torch.Tensor, bound: float = 5.0) -> torch.Tensor:
        return torch.clamp(x, -bound, bound)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(
                self.forget_scale
                * self._clip(self.forget_linear(combined))
                + self.forget_shift
            )
            i = torch.sigmoid(
                self.input_scale
                * self._clip(self.input_linear(combined))
                + self.input_shift
            )
            g = torch.tanh(
                self.update_scale
                * self._clip(self.update_linear(combined))
                + self.update_shift
            )
            o = torch.sigmoid(
                self.output_scale
                * self._clip(self.output_linear(combined))
                + self.output_shift
            )

            # optional quantum‑sampler style stochasticity
            probs = self.sampler_net(hx)
            f = f * probs
            i = i * probs
            g = g * probs
            o = o * probs

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            # modulate hidden state with a classical FCL expectation
            mod = self.fcl(hx).expand_as(hx)
            hx = hx * mod

            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

__all__ = ["QLSTMGen108"]
