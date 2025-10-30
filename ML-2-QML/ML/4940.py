from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, List, Sequence, Callable

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

# Classical Quanvolution filter (from reference pair 4)
class QuanvolutionFilter(nn.Module):
    """Classical 2-D convolutional filter mimicking a quantum kernel."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)

# Lightweight quantum sampler wrapper using Qiskit statevector
class QuantumSamplerWrapper:
    """Wraps a Qiskit circuit to return probabilities via Statevector."""
    def __init__(self, circuit: QuantumCircuit, input_params: ParameterVector, weight_params: ParameterVector):
        self.circuit = circuit
        self.input_params = input_params
        self.weight_params = weight_params

    def run(self, params: Sequence[float], shots: int = 1024) -> np.ndarray:
        """Return probability vector for the computational basis."""
        key_params = self.input_params.params + self.weight_params.params
        mapping = dict(zip(key_params, params))
        bound = self.circuit.assign_parameters(mapping, inplace=False)
        state = Statevector.from_instruction(bound)
        probs = state.probabilities()
        # Ensure order 00,01,10,11
        return np.array([probs[0], probs[1], probs[2], probs[3]], dtype=np.float32)

class HybridSamplerQNN(nn.Module):
    """Hybrid sampler combining classical feature extraction and a quantum sampler.

    Supports two modes:
        *'vector' – 2‑dimensional input processed by a small fully‑connected layer.
        * 'image'  – 28×28 grayscale image processed by a Quanvolution filter.

    The quantum part is a 2‑qubit parameterised circuit that outputs a probability
    distribution over the computational basis.  Optional FastEstimator‑style
    noise injection is available in :meth:`evaluate`.
    """
    def __init__(self, mode: str = "vector", device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)
        self.mode = mode

        # Classical part
        if self.mode == "image":
            self.filter = QuanvolutionFilter()
            self.head = nn.Linear(4 * 14 * 14, 10)
        else:  # vector mode
            self.fc = nn.Linear(2, 4)

        # Quantum part
        self.input_params = ParameterVector("x", 2)
        self.weight_params = ParameterVector("w", 4)
        self._build_circuit()
        self.qsampler = QuantumSamplerWrapper(self.circuit, self.input_params, self.weight_params)

    def _build_circuit(self) -> None:
        self.circuit = QuantumCircuit(2)
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)
        self.circuit.cx(0, 1)
        for w in self.weight_params:
            self.circuit.ry(w, 0 if w.index % 2 == 0 else 1)
        self.circuit.cx(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "image":
            features = self.filter(x)
            logits = self.head(features)
        else:
            hidden = torch.tanh(self.fc(x))
            logits = hidden

        # Prepare parameters for the quantum circuit
        params = torch.cat([x.flatten(), logits.flatten()], dim=0).cpu().numpy()
        probs = self.qsampler.run(params, shots=1024)

        return F.log_softmax(torch.tensor(probs, device=self.device), dim=-1)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Fast estimator with optional Gaussian shot noise."""
        raw: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.tensor(params[:2], dtype=torch.float32, device=self.device)
                outputs = self.forward(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    row.append(float(val))
                raw.append(row)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = [
            [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            for row in raw
        ]
        return noisy

__all__ = ["HybridSamplerQNN"]
