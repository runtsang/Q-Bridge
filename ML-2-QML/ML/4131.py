"""HybridEstimator combines classical and quantum regression models with fast evaluation and optional autoencoding.

The module defines:
- :class:`AutoencoderNet` and :class:`AutoencoderConfig` from the ML autoencoder seed.
- :func:`Autoencoder` factory and :func:`train_autoencoder` training routine.
- :class:`FastBaseEstimator` and :class:`FastEstimator` for deterministic and noisy batch evaluation.
- :class:`HybridEstimator` that accepts either a PyTorch ``nn.Module`` or a Qiskit ``QuantumCircuit``,
  automatically builds a :class:`qiskit_machine_learning.neural_networks.EstimatorQNN` for the quantum branch,
  and exposes a unified ``evaluate`` API.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from typing import Callable, Iterable, List, Sequence, Tuple, Union

# ----------------------------------------------------------------------
# Autoencoder utilities (from reference pair 3)
# ----------------------------------------------------------------------
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    """Simple fully‑connected autoencoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            encoder.append(nn.Linear(in_dim, h))
            encoder.append(nn.ReLU())
            if config.dropout > 0:
                encoder.append(nn.Dropout(config.dropout))
            in_dim = h
        encoder.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            decoder.append(nn.Linear(in_dim, h))
            decoder.append(nn.ReLU())
            if config.dropout > 0:
                decoder.append(nn.Dropout(config.dropout))
            in_dim = h
        decoder.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

def Autoencoder(input_dim: int, *, latent_dim: int = 32,
                hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1) -> AutoencoderNet:
    """Factory that returns an :class:`AutoencoderNet` instance."""
    return AutoencoderNet(AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout))

def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *, epochs: int = 100, batch_size: int = 64,
    lr: float = 1e-3, weight_decay: float = 0.0,
    device: torch.device | None = None
) -> List[float]:
    """Train the autoencoder and return loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

# ----------------------------------------------------------------------
# Helper for converting data to torch tensors
# ----------------------------------------------------------------------
def _as_tensor(data: Union[Iterable[float], torch.Tensor]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

# ----------------------------------------------------------------------
# FastEstimator utilities (from reference pair 2, classical branch)
# ----------------------------------------------------------------------
class FastBaseEstimator:
    """Base class for deterministic batch evaluation of a torch model."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]]
    ) -> List[List[float]]:
        """Deterministic evaluation of observables over parameter sets."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot‑noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy.append([float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row])
        return noisy

# ----------------------------------------------------------------------
# Quantum Estimator utilities (from reference pair 1)
# ----------------------------------------------------------------------
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

def _build_quantum_estimator(
    circuit: QuantumCircuit,
    input_params: Sequence[Parameter],
    weight_params: Sequence[Parameter],
    observables: Sequence["qiskit.quantum_info.operators.base_operator.BaseOperator"]
) -> EstimatorQNN:
    """Instantiate a Qiskit EstimatorQNN with a StatevectorEstimator backend."""
    estimator = StatevectorEstimator()
    return EstimatorQNN(
        circuit=circuit,
        observables=observables,
        input_params=list(input_params),
        weight_params=list(weight_params),
        estimator=estimator,
    )

# ----------------------------------------------------------------------
# HybridEstimator
# ----------------------------------------------------------------------
class HybridEstimator:
    """
    Unified estimator that can work with either a classical PyTorch model or a quantum
    EstimatorQNN.  It supports:

    * Optional autoencoding preprocessing.
    * Fast batch evaluation (deterministic or noisy).
    * Quantum expectation evaluation with a StatevectorEstimator backend.

    Parameters
    ----------
    model : nn.Module | QuantumCircuit
        If a ``nn.Module`` is provided, it will be wrapped by :class:`FastEstimator`.
        If a ``QuantumCircuit`` is provided, a :class:`EstimatorQNN` will be built.
    autoencoder : AutoencoderNet | None
        Optional autoencoder to preprocess inputs before feeding them to the model.
    shots : int | None
        Number of shots for noisy quantum evaluation or Gaussian shot noise for classical.
    seed : int | None
        Random seed for noisy evaluation.
    """

    def __init__(
        self,
        model: Union[nn.Module, QuantumCircuit],
        *,
        autoencoder: AutoencoderNet | None = None,
        shots: int | None = None,
        seed: int | None = None
    ) -> None:
        self.autoencoder = autoencoder
        self.shots = shots
        self.seed = seed

        if isinstance(model, nn.Module):
            self._classical = FastEstimator(model)
            self._quantum = None
        elif isinstance(model, QuantumCircuit):
            # Build a quantum EstimatorQNN; use a dummy observable for regression
            from qiskit.quantum_info.operators import SparsePauliOp
            obs = SparsePauliOp.from_list([("Z" * model.num_qubits, 1)])
            # For simplicity, assume all parameters are weight parameters
            self._quantum = _build_quantum_estimator(
                circuit=model,
                input_params=[],
                weight_params=list(model.parameters()),
                observables=[obs]
            )
            self._classical = None
        else:
            raise TypeError("model must be a torch.nn.Module or qiskit.QuantumCircuit")

    def _preprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.autoencoder is None:
            return inputs
        return self.autoencoder.encode(inputs)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]] | Iterable["qiskit.quantum_info.operators.base_operator.BaseOperator"],
        parameter_sets: Sequence[Sequence[float]]
    ) -> List[List[float]]:
        """
        Evaluate observables over a set of parameter vectors.

        For classical models, ``observables`` should be callables that accept the model output.
        For quantum models, ``observables`` should be Qiskit BaseOperator instances.
        """
        if self._classical is not None:
            return self._classical.evaluate(
                observables=observables,
                parameter_sets=parameter_sets,
                shots=self.shots,
                seed=self.seed
            )
        else:
            # Quantum branch
            # Bind parameters and evaluate using the EstimatorQNN
            results: List[List[float]] = []
            for params in parameter_sets:
                # The EstimatorQNN expects a dict mapping parameters to values
                param_dict = {p: v for p, v in zip(self._quantum.weight_params, params)}
                output = self._quantum(param_dict)
                # Convert complex expectation values to float
                row = [float(np.real(val)) for val in output]
                results.append(row)
            return results

__all__ = [
    "Autoencoder", "AutoencoderConfig", "AutoencoderNet", "train_autoencoder",
    "FastBaseEstimator", "FastEstimator",
    "HybridEstimator"
]
