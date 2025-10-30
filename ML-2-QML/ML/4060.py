import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Callable, List, Sequence, Dict, Any

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class HybridAttentionLayer(nn.Module):
    """Hybrid classical layer combining a fully connected map and a self‑attention block."""
    def __init__(self, input_dim: int, embed_dim: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, 1, bias=True)
        self.embed_dim = embed_dim

    def forward(
        self,
        inputs: torch.Tensor,
        fc_thetas: torch.Tensor,
        attention_rot: torch.Tensor,
        attention_entangle: torch.Tensor,
    ) -> torch.Tensor:
        # Element‑wise scaling of the inputs by the FC parameters
        scaled = inputs * fc_thetas
        fc_out = torch.tanh(self.fc(scaled))

        # Self‑attention
        q = torch.matmul(inputs, attention_rot.reshape(self.embed_dim, -1))
        k = torch.matmul(inputs, attention_entangle.reshape(self.embed_dim, -1))
        scores = torch.softmax(q @ k.T / np.sqrt(self.embed_dim), dim=-1)
        attn_out = torch.matmul(scores, inputs)

        # Combine the two streams
        return fc_out + attn_out.mean(dim=1, keepdim=True)

    def run(
        self,
        params: Dict[str, np.ndarray],
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Convenience wrapper that accepts NumPy arrays."""
        inputs_t = torch.as_tensor(inputs, dtype=torch.float32)
        fc_thetas_t = torch.as_tensor(params["fc_thetas"], dtype=torch.float32)
        attn_rot_t = torch.as_tensor(params["attention_rot"], dtype=torch.float32)
        attn_ent_t = torch.as_tensor(params["attention_entangle"], dtype=torch.float32)
        out = self.forward(inputs_t, fc_thetas_t, attn_rot_t, attn_ent_t)
        return out.detach().numpy()

class FastBaseEstimator:
    """Evaluate the hybrid layer for batches of parameter sets and observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Dict[str, np.ndarray]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                # Assume inputs are embedded in params under key 'inputs'
                inputs = _ensure_batch(params["inputs"])
                out = self.model.forward(
                    inputs,
                    torch.as_tensor(params["fc_thetas"], dtype=torch.float32),
                    torch.as_tensor(params["attention_rot"], dtype=torch.float32),
                    torch.as_tensor(params["attention_entangle"], dtype=torch.float32),
                )
                row: List[float] = []
                for observable in observables:
                    val = observable(out)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Dict[str, np.ndarray]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["HybridAttentionLayer", "FastBaseEstimator", "FastEstimator"]
