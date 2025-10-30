import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence, List, Callable, Tuple

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]],
                          dtype=torch.float32)
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs
    return Layer()

def build_fraud_detection_program(input_params: FraudLayerParameters,
                                  layers: Iterable[FraudLayerParameters]) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Evaluate a PyTorch model for batches of parameters."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(self,
                 observables: Iterable[ScalarObservable],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Wraps FastBaseEstimator and injects Gaussian shot noise."""
    def evaluate(self,
                 observables: Iterable[ScalarObservable],
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

class FCL(nn.Module):
    """A lightweight fully‑connected layer that mimics a parameterised quantum circuit."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

class QLSTM(nn.Module):
    """Quantum‑style LSTM implemented in classical PyTorch."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class HybridFraudQLSTM(nn.Module):
    """
    A hybrid architecture that evaluates a quantum fraud‑detection program
    using FastEstimator and optionally processes the resulting expectation
    values through a classical or quantum LSTM layer.
    """
    def __init__(self,
                 quantum_model: nn.Module,
                 lstm_type: str = "classical",
                 lstm_hidden_dim: int = 32,
                 n_qubits: int = 0,
                 shots: int | None = None,
                 seed: int | None = None) -> None:
        super().__init__()
        # Estimator with optional shot noise
        if shots is None:
            self.estimator = FastBaseEstimator(quantum_model)
        else:
            self.estimator = FastEstimator(quantum_model, shots=shots, seed=seed)

        # Select LSTM backbone
        if lstm_type == "quantum":
            self.lstm = QLSTM(input_dim=1, hidden_dim=lstm_hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden_dim,
                                batch_first=True, num_layers=1)

    def forward(self,
                parameter_sets: Sequence[Sequence[float]]) -> torch.Tensor:
        evs = self.estimator.evaluate([lambda x: x], parameter_sets)
        evs = torch.tensor(evs, dtype=torch.float32).unsqueeze(-1)  # (batch, seq=1, 1)
        if isinstance(self.lstm, QLSTM):
            evs = evs.permute(1, 0, 2)  # (seq, batch, 1)
            outputs, _ = self.lstm(evs)
            outputs = outputs.permute(1, 0, 2)  # (batch, seq, 1)
        else:
            outputs, _ = self.lstm(evs)
        return outputs.squeeze(-1)

    def add_fraud_circuit(self,
                          input_params: FraudLayerParameters,
                          layers: Iterable[FraudLayerParameters]) -> None:
        """Append a fraud‑detection sequence to the underlying model."""
        fraud_circuit = build_fraud_detection_program(input_params, layers)
        self.estimator.model = fraud_circuit

    def add_fcl_layer(self, n_features: int = 1) -> None:
        """Wrap the existing model with an additional FCL quantum‑style layer."""
        fcl_layer = FCL(n_features)
        self.estimator.model = nn.Sequential(fcl_layer, self.estimator.model)

__all__ = ["HybridFraudQLSTM", "FraudLayerParameters", "build_fraud_detection_program", "FCL", "QLSTM"]
