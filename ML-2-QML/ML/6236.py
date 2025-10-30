import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, List, Sequence, Callable, Any

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class QuantumHybridEstimator:
    """
    Classical estimator that can optionally wrap a quantum circuit to
    inject shot noise or compute expectation values on the model output.
    
    Parameters
    ----------
    model : nn.Module
        A PyTorch model that produces a tensor output.
    quantum_circuit : Any, optional
        A quantum circuit object that implements a ``run`` method.
        If provided, the output of the model will be fed to the circuit
        to obtain expectation values.
    shots : int, optional
        Number of shots to use when emulating quantum measurement noise.
    seed : int, optional
        Random seed for reproducibility of shot noise.
    """
    def __init__(self,
                 model: nn.Module,
                 quantum_circuit: Any = None,
                 shots: int | None = None,
                 seed: int | None = None):
        self.model = model
        self.quantum_circuit = quantum_circuit
        self.shots = shots
        self.seed = seed

    def _evaluate_model(self,
                        parameter_sets: Sequence[Sequence[float]]) -> torch.Tensor:
        """Run the PyTorch model on all parameter sets."""
        self.model.eval()
        with torch.no_grad():
            flat_params = np.array(parameter_sets).reshape(-1)
            inputs = _ensure_batch(flat_params)
            outputs = self.model(inputs)
        return outputs

    def _apply_quantum(self, outputs: torch.Tensor) -> torch.Tensor:
        """If a quantum circuit is provided, compute expectation values."""
        if self.quantum_circuit is None:
            return outputs
        values = outputs.squeeze().cpu().numpy()
        expectation = self.quantum_circuit.run(values)
        return torch.as_tensor(expectation, dtype=torch.float32)

    def evaluate(self,
                 observables: Iterable[ScalarObservable] | None = None,
                 parameter_sets: Sequence[Sequence[float]] = [],
                 *,
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[float]]:
        """
        Evaluate the model and optional quantum circuit on a batch of
        parameter sets.  The result is a list of rows, each containing the
        values of the requested observables.
        """
        if shots is None:
            shots = self.shots
        if seed is None:
            seed = self.seed

        outputs = self._evaluate_model(parameter_sets)
        if self.quantum_circuit is not None:
            outputs = self._apply_quantum(outputs)

        if observables is None:
            observables = [lambda out: out.mean(dim=-1)]

        results: List[List[float]] = []
        for out in outputs:
            row: List[float] = []
            for obs in observables:
                val = obs(out)
                if isinstance(val, torch.Tensor):
                    val = float(val.mean().cpu())
                else:
                    val = float(val)
                row.append(val)
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            results = noisy

        return results

__all__ = ["QuantumHybridEstimator"]
